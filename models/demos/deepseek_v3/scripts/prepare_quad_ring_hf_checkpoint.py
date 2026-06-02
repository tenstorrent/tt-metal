#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Export a DeepSeek checkpoint augmented with prepacked quad-ring expert tensors.

This script starts from a stacked dequantized checkpoint (for example
`*-dequantized-stacked`), creates a symlink-based overlay output directory, and adds:
  - model.layers.<L>.mlp.experts_quad_ring.w0_w1.weight
  - model.layers.<L>.mlp.experts_quad_ring.w2.weight

Those tensors are consumed directly by the quad-ring expert conversion path to
skip the expensive on-the-fly repacking step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.demos.deepseek_v3.utils.hf_model_utils import default_quad_ring_model_path, save_quad_ring_hf_checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a stacked DeepSeek checkpoint with prepacked quad-ring expert tensors."
    )
    parser.add_argument(
        "source_model_path",
        type=Path,
        help="Path to the stacked dequantized HF checkpoint directory.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=None,
        help="Output directory for the quad-ring prepared checkpoint. Defaults to '<source>-quad-ring'.",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=128,
        help="Number of devices used to partition experts during prepacking (default: 128 for QUAD 16x8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_model_path = args.output_model_path or default_quad_ring_model_path(args.source_model_path)
    saved_path = save_quad_ring_hf_checkpoint(
        args.source_model_path,
        output_model_path=output_model_path,
        overwrite=args.force,
        num_devices=args.num_devices,
    )
    print(f"Saved quad-ring prepared checkpoint to {saved_path}")


if __name__ == "__main__":
    main()
