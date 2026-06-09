# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load VibeVoice-1.5B and print the model architecture (reference PyTorch)."""

import argparse
import sys
from pathlib import Path

import torch

_REFERENCE_DIR = Path(__file__).resolve().parent
_VIBEVOICE_ROOT = _REFERENCE_DIR.parent
_TT_METAL_ROOT = _VIBEVOICE_ROOT.parent.parent.parent

for path in (_REFERENCE_DIR, _TT_METAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.experimental.vibevoice.common.model_utils import ensure_model_weights  # noqa: E402
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Print VibeVoice-1.5B model architecture")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the local VibeVoice-1.5B weights directory (auto-downloads when omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Device to load the model on",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        args.model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU.")
        args.device = "cpu"

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    device_map = args.device if args.device in ("cuda", "cpu") else None

    print(f"Loading model from {args.model_path} ({args.device}, {dtype})...")
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    if args.device == "mps":
        model.to("mps")

    print(model)


if __name__ == "__main__":
    main()
