#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Export a dequantized DeepSeek HuggingFace checkpoint.

The source checkpoint may contain fp8 tensors together with matching
`*_scale_inv` tensors. This script folds the inverse scales into the weights,
stores bfloat16 tensors, and writes a fresh `model.safetensors.index.json` so
the exported checkpoint can be loaded by the dequantized-only DeepSeek runtime.
"""

import argparse
from pathlib import Path

from models.demos.deepseek_v3.utils.hf_model_utils import default_dequantized_model_path, save_dequantized_hf_checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a dequantized DeepSeek HF checkpoint.")
    parser.add_argument("source_model_path", type=Path, help="Path to the original quantized HF checkpoint directory.")
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=None,
        help="Output directory for the dequantized checkpoint. Defaults to '<source>-dequantized'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_model_path = args.output_model_path or default_dequantized_model_path(args.source_model_path)
    saved_path = save_dequantized_hf_checkpoint(
        args.source_model_path,
        output_model_path=output_model_path,
        overwrite=args.force,
    )
    print(f"Saved dequantized checkpoint to {saved_path}")


if __name__ == "__main__":
    main()
