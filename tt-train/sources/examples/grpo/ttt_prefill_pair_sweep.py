#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill-bucket cross-user KV contamination sweep for Blackhole.

Builds one synthetic prompt of *exact* tokenised length for every entry
in ``--lengths`` (default ``128,1024,4096,8192`` -- one length per
bucket boundary so every pair exercises a distinct bucket combination)
and runs every unordered pair (with self-pairs as baselines) through
the same TTT completer used by ``ttt_gsm8k_debug_example.py``.

Rationale
=========

Real BoolQ prompts are of varying length, so batches end up with a mix
of ``Prefill seq len: 128`` and ``Prefill seq len: 1024`` calls (see
``models/tt_transformers/tt/common.py::get_padded_prefill_len``). The
observed corruption pattern was:

* users routed to the ``128``-bucket kernel produce garbage tokens
  starting near decode position ``128``,
* users routed to the ``1024``-bucket kernel produce clean tokens.

The most economical explanation is that the ``1024`` kernel writes
beyond its own block window in the shared paged-KV cache, leaving stale
data in the pages of any co-batched user whose own bucket didn't fully
overwrite that region.

This script tests that hypothesis directly: it takes every pair
``(L1, L2)`` of target token lengths, runs the two prompts together at
``batch_size=2``, and compares each user's completion to the *self-pair*
baseline (that same length paired with itself, decoded on the same
compiled trace). Any difference is evidence that the batch-mate's
prefill kernel poisoned this user's KV region.

The output is a compact matrix ``[L1][L2] -> "OK" / "USER0 bad" /
"USER1 bad" / "BOTH bad"``. If the KV-cross-contamination hypothesis is
right, exactly the ``(short, long)`` cells where the short user's
``prompt_len + max_new_tokens`` crosses the short bucket boundary should
be bad; ``(short, short)`` and ``(long, long)`` cells should be OK.

Usage
=====

    # Default sweep: 4 lengths (128, 1024, 4096, 8192), one per bucket ->
    #   4 self-pair baselines + C(4,2)=6 mixed pairs = 10 generate() calls.
    python3 ttt_prefill_pair_sweep.py

    # Add more intra-bucket controls (multiple lengths inside 128 and 1024
    # so we can distinguish "different bucket bug" from "different length bug")
    python3 ttt_prefill_pair_sweep.py --lengths 16,64,128,256,1024,4096,8192

    # Longer decode window so slower-to-fail patterns are caught
    python3 ttt_prefill_pair_sweep.py --max-new-tokens 128 --compare-tokens 64
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Sequence, Tuple

# Import the shared machinery from the sibling debug script. Keeping
# these definitions in one place means the sweep speaks the same
# tt-transformers dialect as ``ttt_gsm8k_debug_example.py`` -- same
# ``TttStandaloneCompleter``, same ``ModelOptimizations`` preset, same
# stdout/stderr fd tee, same run-log directory scheme.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from ttt_gsm8k_debug_example import (  # noqa: E402 -- path munging above
    TttStandaloneCompleter,
    _FdTee,
    _resolve_log_dir,
    _has_replacement_chars,
    _DEFAULT_MODEL_ID,
)


# ---------------------------------------------------------------------------
# Synthetic prompt construction
# ---------------------------------------------------------------------------

# Repeated to fill prompts up to the target length. Deliberately chosen to
# be a coherent English sentence so the model has some sensible input
# structure (rather than truly random noise), while being unrelated to any
# Q&A task -- the goal here is *not* to test model quality, it's to test
# whether the batch-mate's prefill kernel corrupts *this* user's KV.
_FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. She sells seashells "
    "by the seashore. All work and no play makes Jack a dull boy. "
)


def _build_prompt_of_length(tokenizer: Any, target_len: int) -> List[int]:
    """Return a token-id list of length exactly ``target_len``.

    Prepends BOS (if the tokenizer has one) so the model sees a
    well-formed start-of-sequence, then pads with repeated
    ``_FILLER_TEXT`` tokens until it hits ``target_len`` exactly.

    We deliberately do NOT apply the chat template: the chat template
    alone eats ~40 tokens on Llama-3.2 (system prompt + role headers),
    which would make lengths 16/32 unrepresentable. Since the goal of
    this sweep is to probe the *prefill kernel*, not the *chat
    behaviour*, raw filler tokens are fine.
    """
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    filler_ids = tokenizer.encode(_FILLER_TEXT, add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no ids for the filler text; cannot build prompts.")
    ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        ids.append(int(tokenizer.bos_token_id))
    while len(ids) < target_len:
        ids.extend(int(t) for t in filler_ids)
    return ids[:target_len]


def _padded_prefill_bucket(prompt_len: int) -> int:
    """Mirror of ``tt_transformers.tt.common.get_padded_prefill_len``."""
    if prompt_len <= 128:
        return 128
    if prompt_len <= 1024:
        return 1024
    return 2 ** (prompt_len - 1).bit_length()


# ---------------------------------------------------------------------------
# Sweep bookkeeping
# ---------------------------------------------------------------------------


def _short_completion(tokenizer: Any, tokens: Sequence[int], limit: int = 12) -> str:
    """Compact one-liner for a completion, decoded with special tokens shown."""
    trimmed = list(tokens)[:limit]
    text = tokenizer.decode(trimmed, skip_special_tokens=False)
    text = " ".join(text.split())  # collapse whitespace
    return text if len(text) <= 80 else text[:79] + "\u2026"


def _completion_status(user_tokens: Sequence[int], baseline_tokens: Sequence[int], compare_len: int) -> str:
    """Return ``"MATCH"`` if the first ``compare_len`` tokens match, else ``"MISMATCH"``.

    Uses ``==`` on the truncated token-id lists. Greedy decoding is
    deterministic in principle, so any mismatch is real evidence the
    batch-mate perturbed this user's state.
    """
    u = list(user_tokens)[:compare_len]
    b = list(baseline_tokens)[:compare_len]
    return "MATCH" if u == b else "MISMATCH"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_ID,
        help=f"HF model id (default: {_DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--lengths",
        default="128,1024,4096,8192",
        help="Comma-separated list of target prompt lengths in tokens "
        "(default: 128,1024,4096,8192 -- one length per bucket boundary "
        "so every pair exercises a distinct bucket combination). Each "
        "length must fit within --max-seq-len minus --max-new-tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens to decode per user (default: 32). Bigger = more chance "
        "to surface downstream corruption; smaller = faster sweep.",
    )
    parser.add_argument(
        "--compare-tokens",
        type=int,
        default=None,
        help="How many leading completion tokens to compare against the "
        "self-pair baseline (default: min(max_new_tokens, 16)).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8320,
        help="Model ``max_seq_len`` (default: 8320 = 8192 + 128, just "
        "enough to fit the default 8192 bucket plus the default 32-token "
        "decode window with headroom. Bump if you enlarge --max-new-tokens "
        "or add larger entries to --lengths.).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy). Greedy makes "
        "the baseline comparison meaningful; if you enable sampling, you "
        "MUST also pass --seed for determinism.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable the captured decode trace. Slower per step but rules " "out trace-side issues.",
    )
    parser.add_argument(
        "--include-self-pairs",
        action="store_true",
        help="Also print the (L, L) baseline rows in the matrix. Off by "
        "default because self-pairs are baselines by construction.",
    )
    parser.add_argument(
        "--dump-completions",
        action="store_true",
        help="Print the decoded completion text for every user on every " "pair (verbose but very readable).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write the run log into (as ``run.log``). "
        "Defaults to ``$TT_METAL_RUNTIME_ROOT/generated/tt-train/ttt_gsm8k_debug_runs/<UTC-timestamp>/``.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable stdout/stderr teeing to a run log (still prints to the terminal).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # OS-fd tee of stdout/stderr -> run.log (must precede any ttnn/loguru
    # import so native prints go through our pipe too). See
    # ``_FdTee.__init__.__doc__`` in the sibling script for details.
    # -----------------------------------------------------------------
    log_file: Optional[BinaryIO] = None
    log_file_path: Optional[Path] = None
    fd_tee: Optional[_FdTee] = None
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "run.log"
        log_file = open(log_file_path, "wb", buffering=0)
        fd_tee = _FdTee(log_file)
        print(f"[log] Writing run log to {log_file_path}", flush=True)
        print(f"[log] argv: {sys.argv}", flush=True)
        print(f"[log] cwd:  {os.getcwd()}", flush=True)

    # Parse and validate lengths.
    try:
        lengths = sorted({int(x.strip()) for x in args.lengths.split(",") if x.strip()})
    except ValueError as exc:
        parser.error(f"--lengths must be a comma-separated list of integers: {exc}")
    if not lengths:
        parser.error("--lengths was empty.")
    if lengths[0] < 1:
        parser.error("--lengths values must be >= 1.")
    if lengths[-1] + args.max_new_tokens > args.max_seq_len:
        parser.error(
            f"largest length ({lengths[-1]}) + max_new_tokens ({args.max_new_tokens}) "
            f"exceeds --max-seq-len ({args.max_seq_len}). Bump --max-seq-len "
            "or drop the largest entry from --lengths."
        )

    compare_len = args.compare_tokens if args.compare_tokens is not None else min(args.max_new_tokens, 16)
    if compare_len < 1:
        parser.error("--compare-tokens must be >= 1.")
    if compare_len > args.max_new_tokens:
        parser.error(f"--compare-tokens ({compare_len}) must be <= --max-new-tokens ({args.max_new_tokens}).")

    # Import ttnn AFTER HF_MODEL is set so anything that reads env at
    # import time sees the right model.
    os.environ["HF_MODEL"] = args.model
    import ttnn

    print("\n=== Sweep configuration ===")
    print(f"Model            : {args.model}")
    print(f"Lengths (tokens) : {lengths}")
    print(f"Buckets          : {sorted({_padded_prefill_bucket(L) for L in lengths})}")
    print(f"max_new_tokens   : {args.max_new_tokens}")
    print(f"compare_tokens   : {compare_len}")
    print(f"max_seq_len      : {args.max_seq_len}")
    print(f"Trace enabled    : {not args.no_trace}")
    print(
        f"Sampling         : temperature={args.temperature}, top_k={args.top_k}, "
        f"top_p={args.top_p}, seed={args.seed}"
    )
    n_pairs = len(lengths) + (len(lengths) * (len(lengths) - 1)) // 2
    print(
        f"Total pair runs  : {n_pairs} ({len(lengths)} self-pairs + "
        f"{(len(lengths) * (len(lengths) - 1)) // 2} mixed pairs)"
    )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    completer: Optional[TttStandaloneCompleter] = None
    try:
        completer = TttStandaloneCompleter(
            mesh_device=mesh_device,
            model_source=args.model,
            max_batch_size=2,
            max_seq_len=args.max_seq_len,
            instruct=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        tokenizer = completer.tokenizer
        print(
            f"\nTokenizer        : {tokenizer.__class__.__name__} "
            f"(vocab={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, "
            f"bos_id={tokenizer.bos_token_id})"
        )

        # Build one prompt per length. Verify exact length was hit.
        prompts: dict[int, List[int]] = {}
        for L in lengths:
            p = _build_prompt_of_length(tokenizer, L)
            assert len(p) == L, f"target length {L} not hit exactly (got {len(p)})"
            prompts[L] = p
            print(f"  length={L:>4}: bucket={_padded_prefill_bucket(L):>4}, " f"first 8 tokens={p[:8]}")

        # -----------------------------------------------------------------
        # 1) Self-pair baselines. We use (L, L) rather than (L, filler)
        #    so the second slot doesn't drag in a different bucket
        #    and change the observed decode -- this way the baseline is
        #    literally "the completion when both users are on the same
        #    bucket". User 0 and user 1 must produce identical output,
        #    which is itself a sanity check that greedy decoding is
        #    deterministic at batch_size=2 with same input.
        # -----------------------------------------------------------------
        print("\n=== Self-pair baselines (batch_size=2, same prompt in both slots) ===")
        baselines: dict[int, List[int]] = {}
        for L in lengths:
            t0 = time.perf_counter()
            completions = completer.generate(
                [prompts[L], prompts[L]],
                max_new_tokens=args.max_new_tokens,
                enable_trace=not args.no_trace,
            )
            elapsed = time.perf_counter() - t0
            u0, u1 = completions[0], completions[1]
            same = u0[:compare_len] == u1[:compare_len]
            baselines[L] = u0
            garbage0 = _has_replacement_chars(tokenizer.decode(u0, skip_special_tokens=True))
            garbage1 = _has_replacement_chars(tokenizer.decode(u1, skip_special_tokens=True))
            marker = "OK" if same and not (garbage0 or garbage1) else "!! CHECK"
            print(
                f"  L={L:>4}  bucket={_padded_prefill_bucket(L):>4}  "
                f"users_agree={same}  garbage0={garbage0}  garbage1={garbage1}  "
                f"{elapsed*1000:.0f}ms  [{marker}]"
            )
            if args.dump_completions:
                print(f"    user0: {_short_completion(tokenizer, u0)}")
                print(f"    user1: {_short_completion(tokenizer, u1)}")

        # -----------------------------------------------------------------
        # 2) Mixed pairs. For each unordered (L1, L2) with L1 < L2,
        #    run once at batch_size=2 and compare against the baselines
        #    of each length in isolation (i.e. the self-pair completion
        #    for that length).
        # -----------------------------------------------------------------
        print("\n=== Mixed pairs ===")
        results: List[Tuple[int, int, str, str, str, bool, bool, List[int], List[int]]] = []
        for L1, L2 in itertools.combinations(lengths, 2):
            t0 = time.perf_counter()
            completions = completer.generate(
                [prompts[L1], prompts[L2]],
                max_new_tokens=args.max_new_tokens,
                enable_trace=not args.no_trace,
            )
            elapsed = time.perf_counter() - t0
            u0, u1 = completions[0], completions[1]
            u0_status = _completion_status(u0, baselines[L1], compare_len)
            u1_status = _completion_status(u1, baselines[L2], compare_len)
            garbage0 = _has_replacement_chars(tokenizer.decode(u0, skip_special_tokens=True))
            garbage1 = _has_replacement_chars(tokenizer.decode(u1, skip_special_tokens=True))
            if u0_status == "MATCH" and u1_status == "MATCH":
                cell = "OK"
            elif u0_status == "MISMATCH" and u1_status == "MISMATCH":
                cell = "BOTH"
            elif u0_status == "MISMATCH":
                cell = "U0_BAD"
            else:
                cell = "U1_BAD"
            results.append((L1, L2, u0_status, u1_status, cell, garbage0, garbage1, u0, u1))
            b1 = _padded_prefill_bucket(L1)
            b2 = _padded_prefill_bucket(L2)
            print(
                f"  L1={L1:>4}(b{b1:>4})  L2={L2:>4}(b{b2:>4})  "
                f"user0={u0_status:<8}  user1={u1_status:<8}  "
                f"garbage0={garbage0}  garbage1={garbage1}  "
                f"{elapsed*1000:.0f}ms  [{cell}]"
            )
            if args.dump_completions or cell != "OK":
                # Always dump the completions on failure so we can spot
                # the corruption pattern (loops, wrong-language tokens,
                # replacement chars, etc.) at a glance.
                print(f"    user0 baseline: {_short_completion(tokenizer, baselines[L1])}")
                print(f"    user0 actual  : {_short_completion(tokenizer, u0)}")
                print(f"    user1 baseline: {_short_completion(tokenizer, baselines[L2])}")
                print(f"    user1 actual  : {_short_completion(tokenizer, u1)}")

        # -----------------------------------------------------------------
        # 3) Compact matrix. Cell (row=L1, col=L2) is populated for the
        #    unordered pair; the transpose cell shows the same string.
        # -----------------------------------------------------------------
        print("\n=== Result matrix ===")
        print("Cell legend:")
        print("  OK      : both users match their self-pair baseline (first " f"{compare_len} tokens)")
        print("  U0_BAD  : the length-L1 user drifted from its baseline")
        print("  U1_BAD  : the length-L2 user drifted from its baseline")
        print("  BOTH    : both drifted")
        print("  self    : diagonal (self-pair; baseline by construction)")

        matrix: dict[Tuple[int, int], str] = {}
        for L1, L2, _u0s, _u1s, cell, *_rest in results:
            matrix[(L1, L2)] = cell
            # Transpose: for the mirrored cell we swap the U0/U1 tag
            # so the cell semantics still refer to the *row's* length.
            mirror = {"OK": "OK", "BOTH": "BOTH", "U0_BAD": "U1_BAD", "U1_BAD": "U0_BAD"}[cell]
            matrix[(L2, L1)] = mirror

        col_headers = "        " + "  ".join(f"L={L:>4}" for L in lengths)
        print(col_headers)
        for L1 in lengths:
            cells = []
            for L2 in lengths:
                if L1 == L2:
                    if args.include_self_pairs:
                        u0 = baselines[L1]
                        cells.append("self  " if u0 else "self  ")
                    else:
                        cells.append("self  ")
                else:
                    cells.append(f"{matrix[(L1, L2)]:<6}")
            print(f"L={L1:>4}  " + "  ".join(cells))

        # -----------------------------------------------------------------
        # 4) Bucket-level summary: aggregate cells by (bucket1, bucket2)
        #    so the "128 vs 1024" story pops out cleanly.
        # -----------------------------------------------------------------
        print("\n=== Bucket-pair summary ===")
        by_bucket: dict[Tuple[int, int], List[str]] = {}
        for L1, L2, _u0s, _u1s, cell, *_rest in results:
            b1 = _padded_prefill_bucket(L1)
            b2 = _padded_prefill_bucket(L2)
            key = tuple(sorted((b1, b2)))
            by_bucket.setdefault(key, []).append(cell)
        for key in sorted(by_bucket.keys()):
            cells = by_bucket[key]
            ok = sum(1 for c in cells if c == "OK")
            bad = len(cells) - ok
            distinct_bads = sorted(set(c for c in cells if c != "OK"))
            print(
                f"  buckets={key}: {len(cells)} pairs, {ok} OK, {bad} bad "
                f"({', '.join(distinct_bads) if distinct_bads else 'clean'})"
            )

        # Overall pass/fail.
        total_bad = sum(1 for (_L1, _L2, _u0s, _u1s, cell, *_r) in results if cell != "OK")
        print(
            f"\nTotal: {len(results)} mixed pairs, {total_bad} corrupted "
            f"({100.0*total_bad/max(1, len(results)):.1f}%)"
        )
        if total_bad == 0:
            print("Result: NO cross-user contamination detected in this sweep.")
        else:
            print(
                "Result: cross-user contamination CONFIRMED. See the matrix "
                "and per-pair dumps above for the exact bucket combinations."
            )
    finally:
        completer = None
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass
        if fd_tee is not None:
            with contextlib.suppress(Exception):
                print(f"[log] Run log written to {log_file_path}", flush=True)
            fd_tee.close()
        if log_file is not None:
            with contextlib.suppress(Exception):
                log_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
