# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DFlash on device: correctness against greedy, plus decode t/s/u.

Two signals, and they are independent, which is what makes this a good first test:

* **Token equality vs non-speculative greedy** validates the *wiring*.  Committed
  tokens are always the target's own argmax, so they must match greedy no matter
  how bad the drafter is - up to the bf16 forward-width noise floor established
  by ``dflash_divergence_probe.py`` (F2).
* **Acceptance rate** validates the *drafter conditioning*.  If the hidden-state
  taps, context assembly or positions are wrong the output is still correct but
  acceptance collapses toward zero.

So: wrong tokens => wiring bug.  Right tokens but ~0 acceptance => conditioning bug.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import DFlashDrafter
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_runner import DFlashRunner
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

PROMPT = "Write a Python function that merges two sorted lists."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--prompts-file",
        default=None,
        help=(
            "one prompt per line; each is run in the SAME process so the 30B target is "
            "loaded once. Acceptance over a single prompt is not a usable statistic (it "
            "spans 2.8-4.0 across equivalent configurations over 11 blocks), so ranking "
            "anything by it needs several prompts at >=40 blocks each."
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--drafter-dtype", default="bfloat8_b", choices=["bfloat8_b", "bfloat16"])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--baseline-trials",
        type=int,
        default=2,
        help="baseline repetitions; the fastest is used, so a cold first decode does not inflate the speedup",
    )
    parser.add_argument(
        "--drafting",
        default="padded",
        choices=["padded", "padded-exact", "incremental"],
        help=(
            "How the drafter is fed its context. padded (default): accumulated prefix "
            "zero-padded to one of seven buckets, so drafter ops hit the ttnn program "
            "cache. incremental: per-iteration delta against a growing K/V cache, which "
            "produces a new shape every iteration and recompiles indefinitely. Over 128 "
            "tokens padded is 120 vs 671 ms per drafter call at IDENTICAL acceptance "
            "(3.05/forward): 12.91 vs 3.93 t/s/u. padded-exact: accumulated path without "
            "padding, kept as the control isolating padding from the rewrite."
        ),
    )
    parser.add_argument(
        "--verify",
        default="aligned",
        choices=["aligned", "from-zero", "decode", "decode_eager"],
        help=(
            "aligned (default): prefill forward restarted at the page-block boundary below "
            "the anchor, so it re-forwards at most page_block_size-1 committed rows. "
            "from-zero: re-forward the whole prefix, O(prefix) and growing. decode / "
            "decode_eager: verify as one batched decode step -- UNSOUND, kept only as a "
            "recorded negative result, because decode rows do not reliably see each "
            "other's same-step K/V writes; see DFlashRunner.verify_mode."
        ),
    )
    parser.add_argument(
        "--page-block",
        type=int,
        default=None,
        help=(
            "paged KV block size. The aligned verify restarts at a multiple of this, so it "
            "sets how many committed rows get re-forwarded: 64 means up to 63, 32 means up "
            "to 31. 32 is the floor -- chunked SDPA needs start_pos to be a multiple of the "
            "32-row tile."
        ),
    )
    parser.add_argument(
        "--no-trace-verify",
        dest="trace_verify",
        action="store_false",
        help=(
            "fall back to the eager verify forward. The traced 32-row window is the "
            "default: 25.2 vs 64.2 ms per verify forward. Kept as an escape hatch because "
            "it is the one path that captures traces mid-generation."
        ),
    )
    parser.add_argument("--verify-width", type=int, default=256, help="padded row count of the from-zero traced verify")
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=32,
        help=(
            "rows in the 32-row traced verify window. Must be 32 to get the win: the "
            "DRAM-sharded decode matmul asserts M == 1, so traced costs 24.48 ms at 32 "
            "rows and 40.99 at 64."
        ),
    )
    parser.add_argument(
        "--offset-free-verify",
        action="store_true",
        help=(
            "capture the verify graph once per generation instead of once per 32-row "
            "window, by moving start_pos into device tensors. Off by default: it runs "
            "without fatals but currently diverges from greedy past the first window."
        ),
    )
    parser.add_argument(
        "--offset-free-eager",
        action="store_true",
        help="diagnostic: run the offset-free graph with no trace, isolating graph from replay",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        logger.info("building target generator ...")
        # The decode verify puts the block's 16 rows in the decode batch, so the batch must
        # be at least block_size.  This costs nothing: the port documents that the decode
        # step always runs 32 rows with inactive ones carrying current_pos = -1, and it
        # measured 70.35 ms at 16 active users against 72.12 ms at 1.
        build_kwargs = {
            "max_batch_size": 32 if args.verify.startswith("decode") else 1,
            "max_seq_len": args.max_seq_len,
        }
        if args.page_block:
            build_kwargs["page_block_size"] = args.page_block
        elif args.trace_verify:
            # The traced window is 32 rows and must start on a page-block boundary, so the
            # block size has to divide it. 32 is also the floor the chunked SDPA offset can
            # shrink to, so start_pos stays legal for every anchor.
            build_kwargs["page_block_size"] = 32
        gen = build_generator(".", mesh, **build_kwargs)
        tok = gen.tokenizer

        if args.prompts_file:
            prompts = [line.strip() for line in open(args.prompts_file) if line.strip()]
        else:
            prompts = [args.prompt]
        logger.info(f"{len(prompts)} prompt(s)")

        logger.info(f"building drafter (weights {args.drafter_dtype}) ...")
        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=R.draft_config(),
            mesh_device=mesh,
            weight_dtype=getattr(ttnn, args.drafter_dtype),
            activation_dtype=ttnn.bfloat16,
        )
        runner = DFlashRunner(
            gen,
            drafter,
            padded_drafting=args.drafting in ("padded", "padded-exact"),
            pad_context=args.drafting == "padded",
            aligned_verify=args.verify != "from-zero",
            verify_mode=args.verify if args.verify.startswith("decode") else "prefill",
            trace_verify=args.trace_verify,
            verify_width=args.verify_width,
            verify_rows=args.verify_rows,
            offset_free_verify=args.offset_free_verify,
            offset_free_eager=args.offset_free_eager,
        )

        per_prompt = []
        for index, prompt_text in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tok(text)["input_ids"]
            logger.info(f"=== prompt {index}: {len(prompt_ids)} tokens ===")
            gen.reset()
            dflash_tokens, stats = runner.generate(prompt_ids, args.max_new_tokens)
            logger.info(json.dumps(stats.as_dict(), indent=2))
            per_prompt.append(
                {
                    "prompt": prompt_text,
                    "prompt_tokens": len(prompt_ids),
                    "dflash": stats.as_dict(),
                    "dflash_token_ids": dflash_tokens,
                }
            )

        # -------------------------------------------------------- baseline
        baseline_tokens: list[int] = []
        baseline_seconds = 0.0
        baseline_trials: list[float] = []
        if not args.skip_baseline:
            logger.info("=== baseline greedy (no speculation) ===")
            # Run it more than once and keep the FASTEST.  A single trial here measured
            # 32.5 t/s/u against 42.9 measured directly, because the first decode of a
            # fresh generator pays trace capture and program-cache population. Comparing
            # a warm DFlash number against a cold baseline would flatter DFlash by ~30 %.
            for trial in range(max(1, args.baseline_trials)):
                gen.reset()
                started = time.perf_counter()
                baseline_tokens = gen.generate(prompt_ids, args.max_new_tokens)
                elapsed = time.perf_counter() - started
                baseline_trials.append(elapsed)
                logger.info(f"baseline trial {trial}: {elapsed:.3f}s " f"({len(baseline_tokens) / elapsed:.2f} t/s/u)")
            baseline_seconds = min(baseline_trials)

        n = min(len(baseline_tokens), len(dflash_tokens)) if baseline_tokens else 0
        mismatches = [i for i in range(n) if baseline_tokens[i] != dflash_tokens[i]]

        baseline_tps = len(baseline_tokens) / baseline_seconds if baseline_seconds else 0.0
        speedup = stats.tokens_per_second / baseline_tps if baseline_tps else 0.0

        print("\n" + "=" * 72)
        print(f"prompt tokens            : {len(prompt_ids)}")
        print(f"DFlash tokens            : {stats.tokens}")
        print(f"DFlash iterations        : {stats.iterations}")
        print(f"target forwards          : {stats.target_forwards}")
        print(f"accepted / target forward: {stats.accepted_per_target_forward:.2f}  (ceiling 16)")
        print(f"mean matches per block   : {stats.mean_matches:.2f}  of 15")
        print(f"per-block matches        : {stats.matches}")
        print(f"draft / verify seconds   : {stats.draft_seconds:.2f} / {stats.verify_seconds:.2f}")
        print(f"DFlash t/s/u             : {stats.tokens_per_second:.2f}   ({stats.ms_per_token:.2f} ms/token)")
        if baseline_tokens:
            print(f"baseline t/s/u           : {baseline_tps:.2f}")
            print(f"SPEEDUP                  : {speedup:.2f}x")
            print(f"token mismatches vs greedy: {len(mismatches)} / {n}  at {mismatches[:8]}")
        print("=" * 72)
        if len(per_prompt) > 1:
            # Acceptance is the statistic that decides whether speculation can win at
            # all, and it is far too noisy to read off one prompt -- so report the
            # spread, not just a mean.
            print("\nper-prompt acceptance (the metric that decides viability):")
            print(f"  {'prompt':>6s} {'blocks':>7s} {'matches/block':>14s} {'accepted/fwd':>13s} {'t/s/u':>8s}")
            for i, entry in enumerate(per_prompt):
                d = entry["dflash"]
                print(
                    f"  {i:>6d} {d['iterations']:>7d} {d['mean_matches']:>14.2f} "
                    f"{d['accepted_per_target_forward']:>13.2f} {d['tokens_per_second']:>8.2f}"
                )
            blocks = sum(d["dflash"]["iterations"] for d in per_prompt)
            total_matches = sum(sum(d["dflash"]["matches"]) for d in per_prompt)
            values = [d["dflash"]["mean_matches"] for d in per_prompt]
            print(
                f"  pooled over {blocks} blocks: {total_matches / blocks:.2f} matches/block of 15 "
                f"(per-prompt range {min(values):.2f}-{max(values):.2f})"
            )
        print("\nDFlash text:\n" + tok.decode(dflash_tokens, skip_special_tokens=True))

        payload = {
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_ids),
            "drafter_weight_dtype": args.drafter_dtype,
            "drafting": args.drafting,
            "page_block": args.page_block,
            "verify": args.verify,
            "trace_verify": args.trace_verify,
            "dflash": stats.as_dict(),
            "dflash_token_ids": dflash_tokens,
            "baseline_token_ids": baseline_tokens,
            "baseline_seconds": baseline_seconds,
            "baseline_trial_seconds": baseline_trials,
            "baseline_tokens_per_second": baseline_tps,
            "speedup": speedup,
            "token_mismatch_count": len(mismatches),
            "token_mismatch_indices": mismatches,
            "tokens_compared": n,
            "per_prompt": per_prompt,
        }
        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_device_e2e.json")
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
