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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        logger.info("building target generator ...")
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=args.max_seq_len)
        tok = gen.tokenizer

        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok(text)["input_ids"]
        logger.info(f"prompt_len={len(prompt_ids)}")

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
        )

        # ---------------------------------------------------------- DFlash
        logger.info("=== DFlash ===")
        dflash_tokens, stats = runner.generate(prompt_ids, args.max_new_tokens)
        logger.info(json.dumps(stats.as_dict(), indent=2))

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
        print("\nDFlash text:\n" + tok.decode(dflash_tokens, skip_special_tokens=True))

        payload = {
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_ids),
            "drafter_weight_dtype": args.drafter_dtype,
            "drafting": args.drafting,
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
        }
        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_device_e2e.json")
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
