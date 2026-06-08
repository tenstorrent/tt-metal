#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo interactive server — REPL for generating images from text prompts.

Uses Metal Trace on all three models (TE, DIT, VAE) for fast inference.

Usage:
    python -m models.demos.z_image_turbo.demo.demo_server
    python -m models.demos.z_image_turbo.demo.demo_server --steps 9 --seed 42
"""

import argparse
import os
import time

from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo

DEFAULT_STEPS = 9
DEFAULT_SEED = 42
OUTPUT_DIR = "outputs"
WARMUP_PROMPT = "a cat sitting on a mat"


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo interactive server with Metal Trace",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Denoising steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED}); " "each prompt uses seed + idx so repeats yield new images",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import readline  # noqa: F401  -- enables history/editing inside input()
    except ImportError:
        pass  # readline unavailable; continue without line-editing support

    # 1) Load models + warmup (compile all programs + capture TE, DIT & VAE traces).
    pipeline = ZImageTurbo()

    bar = "=" * 72
    print(bar)
    print(f"Warming up with dummy prompt {WARMUP_PROMPT!r}")
    print("(first run is slow: compile all programs + TE, DIT & VAE Metal Trace capture)")
    print(bar)

    t0 = time.time()
    warmup_image = pipeline.warmup(steps=args.steps, seed=args.seed)

    path = os.path.join(OUTPUT_DIR, "out_0.png")
    warmup_image.save(path)
    print(f"  END-TO-END: {time.time() - t0:.2f} s  |  {os.path.abspath(path)}\n")

    # 2) REPL loop.
    print(bar)
    print("Ready. Type a prompt and press ENTER to generate.")
    print("Ctrl-D or Ctrl-C to exit.")
    print(bar)

    idx = 1
    while True:
        try:
            prompt = input(f"\nprompt [{idx}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue

        try:
            path = os.path.join(OUTPUT_DIR, f"out_{idx}.png")
            t_start = time.time()
            image = pipeline(
                prompt,
                steps=args.steps,
                seed=args.seed + idx,
            )
            image.save(path)
            elapsed = time.time() - t_start
            print(f"  END-TO-END: {elapsed:.2f} s  |  {os.path.abspath(path)}\n")
            idx += 1
        except KeyboardInterrupt:
            print("\n  Interrupted during generation; shutting down.")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


if __name__ == "__main__":
    main()
