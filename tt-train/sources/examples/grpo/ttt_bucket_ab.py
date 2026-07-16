#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal two-length bucket A/B test on Blackhole.

Given two customisable prompt lengths ``L1`` and ``L2`` (default
``128`` and ``1024``), this script does exactly three ``generate()``
calls through the tt-transformers stack:

    1. batch=[L1]        -- batch_size=1, one user of length L1
    2. batch=[L2]        -- batch_size=1, one user of length L2
    3. batch=[L1, L2]    -- batch_size=2, both prompts together

and prints the completion for every user in every call. It then
compares user 0 of the mixed batch against the single-user result for
L1, and user 1 of the mixed batch against the single-user result for
L2. Any mismatch is direct evidence that co-batching with the other
length changed this user's output.

Compared to ``ttt_prefill_pair_sweep.py``:

* No self-pair baselines -- the alone-batches ARE the baselines,
  measured with ``max_batch_size=1`` so there's no filler user at all.
* Two separate completer instances (one at ``max_batch_size=1``, one at
  ``max_batch_size=2``) so the alone-batch traces and the paired-batch
  trace don't interfere with each other. Both share the same
  ``mesh_device``.

Usage
=====

    export TT_METAL_HOME=/localdev/ichovpan/tt-metal
    export HF_TOKEN=<your token>
    cd $TT_METAL_HOME/tt-train/sources/examples/grpo

    # Default: L1=128, L2=1024
    python3 ttt_bucket_ab.py

    # Custom lengths
    python3 ttt_bucket_ab.py --l1 64 --l2 4096

    # Bigger decode window if you want to see downstream corruption grow
    python3 ttt_bucket_ab.py --max-new-tokens 128 --compare-tokens 64
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Sequence

# Reuse the completer, log-file tee, and log-dir resolution from the
# sibling debug script -- same tt-transformers dialect, same paged
# attention sizing, same on-device sampling contract.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from ttt_gsm8k_debug_example import (  # noqa: E402 -- path munging above
    TttStandaloneCompleter,
    _FdTee,
    _resolve_log_dir,
    _DEFAULT_MODEL_ID,
)


# Same filler text as the pair sweep, so any comparison between the two
# scripts uses matching prompt content.
_FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. She sells seashells "
    "by the seashore. All work and no play makes Jack a dull boy. "
)


def _build_prompt_of_length(tokenizer: Any, target_len: int) -> List[int]:
    """Return a token-id list of length exactly ``target_len``.

    Prepends BOS (if the tokenizer has one) then pads with repeated
    filler tokens. Same approach as ``ttt_prefill_pair_sweep.py`` so
    outputs are byte-for-byte comparable between the two scripts.
    """
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    filler_ids = tokenizer.encode(_FILLER_TEXT, add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no ids for the filler text.")
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


def _full_text(tokenizer: Any, tokens: Sequence[int]) -> str:
    """Full decoded text of a completion, special tokens stripped.

    Used for the human-readable side-by-side view. We collapse
    whitespace (so multi-line garbage outputs still line up cleanly
    under the batch labels) and preserve everything else verbatim.
    """
    text = tokenizer.decode(list(tokens), skip_special_tokens=True)
    return " ".join(text.split())


def _preview_text(tokenizer: Any, tokens: Sequence[int], limit: int) -> str:
    """First ``limit`` tokens decoded with special tokens preserved.

    Special tokens are kept so the ``<|eot_id|>`` at the end of a
    normal completion is visible (its early appearance is a normal
    stop, its absence is suspicious).
    """
    trimmed = list(tokens)[:limit]
    text = tokenizer.decode(trimmed, skip_special_tokens=False)
    text = " ".join(text.split())
    return text if len(text) <= 1000 else text[:999] + "\u2026"


def _fmt_ids(ids: Sequence[int], limit: int = 16) -> str:
    """Comma-joined token-id preview for at-a-glance diffing."""
    shown = list(ids)[:limit]
    tail = "" if len(ids) <= limit else f", ... {len(ids) - limit} more"
    return "[" + ", ".join(str(int(t)) for t in shown) + tail + "]"


def _first_divergence(alone: Sequence[int], paired: Sequence[int]) -> Optional[tuple]:
    """Return ``(index, alone_tok, paired_tok)`` of the first differing token.

    Returns ``None`` if the two sequences are identical up to the
    length of the shorter one AND the same length. If one is a strict
    prefix of the other, returns ``(shorter_len, None, None)``.
    """
    for i, (a, b) in enumerate(zip(alone, paired)):
        if int(a) != int(b):
            return (i, int(a), int(b))
    if len(alone) != len(paired):
        return (min(len(alone), len(paired)), None, None)
    return None


def _side_by_side(
    label: str,
    alone_tokens: Sequence[int],
    paired_tokens: Sequence[int],
    tokenizer: Any,
) -> None:
    """Print one section of the summary: full text, then divergence."""
    print(label)
    print("-" * 80)
    print(f"  batch_size=1 (alone) : {_full_text(tokenizer, alone_tokens)}")
    print(f"  batch_size=2 (paired): {_full_text(tokenizer, paired_tokens)}")
    div = _first_divergence(alone_tokens, paired_tokens)
    if div is None:
        print("  first divergence     : MATCH (byte-identical)")
    else:
        idx, a_tok, b_tok = div
        if a_tok is None or b_tok is None:
            print(
                f"  first divergence     : one sequence ends at token {idx} "
                f"(alone_len={len(alone_tokens)}, paired_len={len(paired_tokens)})"
            )
        else:
            a_piece = tokenizer.decode([a_tok], skip_special_tokens=False)
            b_piece = tokenizer.decode([b_tok], skip_special_tokens=False)
            print(
                f"  first divergence     : token #{idx}  "
                f"alone=id{a_tok} {a_piece!r}  vs  paired=id{b_tok} {b_piece!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_ID,
        help=f"HF model id (default: {_DEFAULT_MODEL_ID}).",
    )
    parser.add_argument("--l1", type=int, default=128, help="First prompt length in tokens (default: 128).")
    parser.add_argument("--l2", type=int, default=1024, help="Second prompt length in tokens (default: 1024).")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens to decode per user (default: 32).",
    )
    parser.add_argument(
        "--compare-tokens",
        type=int,
        default=None,
        help="Leading tokens compared between alone and paired completions " "(default: min(max_new_tokens, 16)).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Model ``max_seq_len``. Defaults to ``max(l1, l2) + max(max_new_tokens, 128)`` "
        "-- just enough to fit the largest prompt plus the decode window.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable captured decode trace on both completers.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write the run log into (as ``run.log``). Defaults to "
        "``$TT_METAL_RUNTIME_ROOT/generated/tt-train/ttt_gsm8k_debug_runs/<UTC-timestamp>/``.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable stdout/stderr teeing to a run log.")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # OS-fd tee of stdout/stderr -> run.log. Must be set up BEFORE
    # anything ttnn / loguru prints (see _FdTee.__init__.__doc__ in the
    # sibling debug script for details).
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

    if args.l1 < 1 or args.l2 < 1:
        parser.error("--l1 and --l2 must both be >= 1.")

    compare_tokens = args.compare_tokens if args.compare_tokens is not None else min(args.max_new_tokens, 16)
    if compare_tokens < 1 or compare_tokens > args.max_new_tokens:
        parser.error(f"--compare-tokens must be in [1, {args.max_new_tokens}] (got {compare_tokens})")

    largest = max(args.l1, args.l2)
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    else:
        # Fit the largest prompt AND its decode window with at least
        # 128 tokens of slack. Just ``largest + 128`` isn't enough when
        # --max-new-tokens is bumped past 128 (e.g. 2048 + 1024 = 3072
        # would blow through the naive 2048+128 default).
        max_seq_len = largest + max(args.max_new_tokens, 128)
    if largest + args.max_new_tokens > max_seq_len:
        parser.error(
            f"max(--l1,--l2)={largest} + --max-new-tokens={args.max_new_tokens} "
            f"exceeds --max-seq-len={max_seq_len}. Increase --max-seq-len."
        )

    os.environ["HF_MODEL"] = args.model
    import ttnn

    print("\n=== Configuration ===")
    print(f"Model         : {args.model}")
    print(f"L1            : {args.l1} tokens  (bucket {_padded_prefill_bucket(args.l1)})")
    print(f"L2            : {args.l2} tokens  (bucket {_padded_prefill_bucket(args.l2)})")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"compare_tokens: {compare_tokens}")
    print(f"max_seq_len   : {max_seq_len}")
    print(f"Trace enabled : {not args.no_trace}")
    print(
        f"Sampling      : temperature={args.temperature}, top_k={args.top_k}, " f"top_p={args.top_p}, seed={args.seed}"
    )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    completer_single: Optional[TttStandaloneCompleter] = None
    completer_double: Optional[TttStandaloneCompleter] = None
    try:
        # Two independent completer instances share the same mesh_device
        # but each carries its own model weights, KV cache, and captured
        # decode trace. This is the cleanest way to run a real
        # batch_size=1 call (no filler user at all) alongside a real
        # batch_size=2 call.
        print("\n=== Building completer (max_batch_size=1) for alone-batches ===")
        t0 = time.perf_counter()
        completer_single = TttStandaloneCompleter(
            mesh_device=mesh_device,
            model_source=args.model,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            instruct=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(f"    built in {time.perf_counter() - t0:.1f}s")

        print("\n=== Building completer (max_batch_size=2) for the paired batch ===")
        t0 = time.perf_counter()
        completer_double = TttStandaloneCompleter(
            mesh_device=mesh_device,
            model_source=args.model,
            max_batch_size=2,
            max_seq_len=max_seq_len,
            instruct=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(f"    built in {time.perf_counter() - t0:.1f}s")

        tokenizer = completer_single.tokenizer
        print(
            f"\nTokenizer     : {tokenizer.__class__.__name__} "
            f"(vocab={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, "
            f"bos_id={tokenizer.bos_token_id})"
        )

        # Build the two prompts. Assert exact length so the bucket
        # decision in the framework matches what we advertise.
        prompt_l1 = _build_prompt_of_length(tokenizer, args.l1)
        prompt_l2 = _build_prompt_of_length(tokenizer, args.l2)
        assert len(prompt_l1) == args.l1
        assert len(prompt_l2) == args.l2

        # -----------------------------------------------------------------
        # Print the exact inputs the model will see -- the FULL decoded
        # text of both prompts, no truncation. For synthetic filler this
        # will be the same repeated sentence, but the length + bucket
        # labels make it obvious which prompt maps to which prefill
        # kernel. For >2048 token prompts this may print a lot; the log
        # file captures it either way so you can always search back.
        # -----------------------------------------------------------------
        print("\n=== Prompts fed to the model ===")
        prompt_preview_ids = 16
        print(f"\nPrompt A (L1={args.l1} tokens, bucket {_padded_prefill_bucket(args.l1)}):")
        print(f"    first {prompt_preview_ids} ids: {_fmt_ids(prompt_l1, prompt_preview_ids)}")
        print(f"    text:")
        print(tokenizer.decode(prompt_l1, skip_special_tokens=False))
        print(f"\nPrompt B (L2={args.l2} tokens, bucket {_padded_prefill_bucket(args.l2)}):")
        print(f"    first {prompt_preview_ids} ids: {_fmt_ids(prompt_l2, prompt_preview_ids)}")
        print(f"    text:")
        print(tokenizer.decode(prompt_l2, skip_special_tokens=False))

        # -----------------------------------------------------------------
        # 1) batch=[L1]: batch_size=1, one user of length L1.
        # -----------------------------------------------------------------
        print(f"\n=== Batch A: [L1={args.l1}] alone (batch_size=1) ===")
        t0 = time.perf_counter()
        completions_a = completer_single.generate(
            [prompt_l1],
            max_new_tokens=args.max_new_tokens,
            enable_trace=not args.no_trace,
        )
        alone_l1 = completions_a[0]
        print(f"    elapsed  : {time.perf_counter() - t0:.2f}s")
        print(f"    tokens   : {len(alone_l1)}  ids[0:{compare_tokens}] = {_fmt_ids(alone_l1, compare_tokens)}")
        print(f"    full text: {_full_text(tokenizer, alone_l1)}")

        # -----------------------------------------------------------------
        # 2) batch=[L2]: batch_size=1, one user of length L2.
        # -----------------------------------------------------------------
        print(f"\n=== Batch B: [L2={args.l2}] alone (batch_size=1) ===")
        t0 = time.perf_counter()
        completions_b = completer_single.generate(
            [prompt_l2],
            max_new_tokens=args.max_new_tokens,
            enable_trace=not args.no_trace,
        )
        alone_l2 = completions_b[0]
        print(f"    elapsed  : {time.perf_counter() - t0:.2f}s")
        print(f"    tokens   : {len(alone_l2)}  ids[0:{compare_tokens}] = {_fmt_ids(alone_l2, compare_tokens)}")
        print(f"    full text: {_full_text(tokenizer, alone_l2)}")

        # -----------------------------------------------------------------
        # 3) batch=[L1, L2]: batch_size=2, both prompts in one call.
        # -----------------------------------------------------------------
        print(f"\n=== Batch C: [L1={args.l1}, L2={args.l2}] together (batch_size=2) ===")
        t0 = time.perf_counter()
        completions_c = completer_double.generate(
            [prompt_l1, prompt_l2],
            max_new_tokens=args.max_new_tokens,
            enable_trace=not args.no_trace,
        )
        paired_l1, paired_l2 = completions_c[0], completions_c[1]
        print(f"    elapsed  : {time.perf_counter() - t0:.2f}s")
        print(
            f"    user0 (L1={args.l1}) tokens   : {len(paired_l1)}  ids[0:{compare_tokens}] = {_fmt_ids(paired_l1, compare_tokens)}"
        )
        print(f"    user0 (L1={args.l1}) full text: {_full_text(tokenizer, paired_l1)}")
        print(
            f"    user1 (L2={args.l2}) tokens   : {len(paired_l2)}  ids[0:{compare_tokens}] = {_fmt_ids(paired_l2, compare_tokens)}"
        )
        print(f"    user1 (L2={args.l2}) full text: {_full_text(tokenizer, paired_l2)}")

        # -----------------------------------------------------------------
        # Side-by-side summary. This is the section a human should scan
        # first: for each length it stacks the alone (batch_size=1) and
        # paired (batch_size=2) outputs on adjacent lines, prints the
        # first divergence, and finally a verdict block.
        # -----------------------------------------------------------------
        l1_match = alone_l1[:compare_tokens] == paired_l1[:compare_tokens]
        l2_match = alone_l2[:compare_tokens] == paired_l2[:compare_tokens]

        print("\n" + "=" * 80)
        print("=== SIDE-BY-SIDE: alone (batch_size=1) vs paired (batch_size=2) ===")
        print("=" * 80)
        _side_by_side(
            f"\nL1 = {args.l1} tokens (bucket {_padded_prefill_bucket(args.l1)})",
            alone_l1,
            paired_l1,
            tokenizer,
        )
        _side_by_side(
            f"\nL2 = {args.l2} tokens (bucket {_padded_prefill_bucket(args.l2)})",
            alone_l2,
            paired_l2,
            tokenizer,
        )

        print("\n=== Verdict ===")
        print(f"  L1={args.l1:>5}  first {compare_tokens} tokens: " f"{'MATCH' if l1_match else 'MISMATCH'}")
        print(f"  L2={args.l2:>5}  first {compare_tokens} tokens: " f"{'MATCH' if l2_match else 'MISMATCH'}")
        if l1_match and l2_match:
            print(
                "\nBoth users produce identical output whether alone or paired. "
                "No cross-user contamination detected in this pair.\n"
                "If you expected contamination, remember decode has to CROSS the "
                "shorter bucket's boundary to sample poisoned KV pages -- "
                "try increasing --max-new-tokens so prompt_len + max_new_tokens > "
                f"{_padded_prefill_bucket(min(args.l1, args.l2))}."
            )
        else:
            corrupted = []
            if not l1_match:
                corrupted.append(f"L1={args.l1} (bucket {_padded_prefill_bucket(args.l1)})")
            if not l2_match:
                corrupted.append(f"L2={args.l2} (bucket {_padded_prefill_bucket(args.l2)})")
            print(
                "\nCross-user contamination CONFIRMED. Users that drifted when " f"co-batched: {', '.join(corrupted)}."
            )
    finally:
        completer_single = None
        completer_double = None
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
