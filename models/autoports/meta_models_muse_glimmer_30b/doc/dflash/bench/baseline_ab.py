"""Non-speculative decode t/s/u and TTFT, for A/B-ing two checkouts of the model.

Kept OUTSIDE the repository on purpose: it has to run unmodified against the pristine
pre-DFlash commit and against the DFlash tip, so it cannot live in either tree.  Point
PYTHONPATH at the worktree under test; everything else (interpreter, tt-metal build,
device) is held fixed.

It measures the path DFlash is compared against -- plain greedy decode with no
speculation -- and reports the fastest of N trials, because a fresh generator's first
decode pays trace capture and program-cache population and would otherwise read ~30%
slow.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from loguru import logger

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

PROMPT = "Write a Python function that merges two sorted lists."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import models.autoports.meta_models_muse_glimmer_30b as package

    root = os.path.realpath(list(package.__path__)[0])
    logger.info(f"model code: {root}")

    mesh = open_generator_mesh()
    try:
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=args.max_seq_len)
        tok = gen.tokenizer
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = list(tok(text)["input_ids"])

        totals: list[float] = []
        ttfts: list[float] = []
        tokens_out = None
        for trial in range(args.trials):
            gen.reset()
            started = time.perf_counter()
            out = gen.generate(prompt_ids, args.max_new_tokens)
            elapsed = time.perf_counter() - started
            totals.append(elapsed)
            tokens_out = list(out)

            # TTFT on its own: one token from a fresh state.
            gen.reset()
            t0 = time.perf_counter()
            gen.generate(prompt_ids, 1)
            ttfts.append(time.perf_counter() - t0)
            logger.info(
                f"trial {trial}: {elapsed:.3f}s  {len(out) / elapsed:.2f} t/s/u   ttft {1000 * ttfts[-1]:.1f} ms"
            )

        best = min(totals)
        best_ttft = min(ttfts)
        result = {
            "label": args.label,
            "model_root": root,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": args.max_new_tokens,
            "trial_seconds": totals,
            "best_seconds": best,
            "tokens_per_second": args.max_new_tokens / best,
            "ms_per_token": 1000.0 * best / args.max_new_tokens,
            "ttft_seconds": ttfts,
            "best_ttft_ms": 1000.0 * best_ttft,
            "median_tokens_per_second": args.max_new_tokens / statistics.median(totals),
            "first_tokens": tokens_out[:16] if tokens_out else [],
        }
        print("\n" + "=" * 72)
        print(f"label          : {args.label}")
        print(f"model code     : {root}")
        print(f"decode t/s/u   : {result['tokens_per_second']:.2f}   ({result['ms_per_token']:.2f} ms/token)")
        print(f"TTFT           : {result['best_ttft_ms']:.1f} ms")
        print(f"first 16 tokens: {result['first_tokens']}")
        print("=" * 72)

        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
            print(f"wrote {args.out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
