#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo demo — generate 512x512 images from text prompts.

Hardware: 2x Blackhole P300 (QB2), tensor-parallel across (1,4) mesh.
Models:   text encoder + transformer + VAE decoder all on TTNN with Metal Trace.

Single prompt:
    python -m models.demos.z_image_turbo.demo.demo "a misty mountain lake at dawn"
    python -m models.demos.z_image_turbo.demo.demo "a robot in an art studio" --output robot.png --seed 0

Multiple prompts (loaded once, run back to back):
    python -m models.demos.z_image_turbo.demo.demo "a cat" "a dog" "a fox"
    python -m models.demos.z_image_turbo.demo.demo "a cat" "a dog" --output-dir results/ --seed 42 --steps 9
    python -m models.demos.z_image_turbo.demo.demo --prompts-file prompts.txt --output-dir results/ --seed 42

prompts.txt format: one prompt per line, blank lines and # comments ignored.
"""

import argparse
import os
import re
import time

from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo

# ── Output filename ────────────────────────────────────────────────────────────


def _output_path(prompt, index, output_dir):
    """Build an output filename from the prompt and index."""
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")[:40]
    return os.path.join(output_dir, f"{index:02d}_{slug}.png")


# ── Public API ─────────────────────────────────────────────────────────────────


def run(prompts, steps=9, seed=42, output_dir=".", output=None):
    """Generate images for one or more prompts, loading models once.

    Args:
        prompts:    list of prompt strings.
        steps:      denoising steps (same for all prompts).
        seed:       random seed used for all prompts.
        output_dir: directory for output files when len(prompts) > 1.
        output:     explicit output path; only used when len(prompts) == 1.

    Returns:
        List of output file paths.
    """
    if not prompts:
        raise ValueError("No prompts provided.")

    pipeline = ZImageTurbo()

    bar = "=" * 72
    print(bar)
    print("Warming up (compile all programs + capture Metal Traces) ...")
    print(bar)
    t0 = time.time()
    _ = pipeline.warmup(steps=steps, seed=seed)
    print(f"Warmup done in {time.time() - t0:.1f} s\n")

    t_wall = time.time()
    outputs = []

    for i, prompt in enumerate(prompts, start=1):
        if len(prompts) == 1 and output:
            path = output
        else:
            path = _output_path(prompt, i, output_dir)

        header = f"[{i}/{len(prompts)}]" if len(prompts) > 1 else ""
        print(f"{header} Prompt: {prompt!r}  (seed={seed}, steps={steps})")

        t_start = time.time()
        image = pipeline(prompt, steps=steps, seed=seed)
        elapsed = time.time() - t_start

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        image.save(path)
        print(f"  → {path}  ({elapsed:.2f} s)\n")
        outputs.append(path)

    if len(prompts) > 1:
        print(f"{'─' * 60}")
        print(f"All {len(prompts)} images done in {(time.time() - t_wall):.1f} s")
        print(f"Outputs: {output_dir}/")

    return outputs


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo — text-to-image on 2x Blackhole P300 (QB2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m models.demos.z_image_turbo.demo.demo "a cat"
  python -m models.demos.z_image_turbo.demo.demo "a cat" "a dog" "a fox" --output-dir results/
  python -m models.demos.z_image_turbo.demo.demo --prompts-file prompts.txt --output-dir results/ --steps 9
""",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="One or more text prompts (positional)",
    )
    parser.add_argument(
        "--prompts-file",
        metavar="FILE",
        help="Text file with one prompt per line (# comments and blank lines ignored)",
    )
    parser.add_argument(
        "--output",
        default="output.png",
        help="Output file path for a single prompt (default: output.png)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for multiple prompts (default: current dir)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Denoising steps (default: 9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42); same seed used for all prompts",
    )
    args = parser.parse_args()

    prompts = list(args.prompts)

    if args.prompts_file:
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)

    if not prompts:
        parser.error("Provide at least one prompt (positional) or --prompts-file.")

    run(
        prompts=prompts,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        output=args.output if len(prompts) == 1 else None,
    )


if __name__ == "__main__":
    main()
