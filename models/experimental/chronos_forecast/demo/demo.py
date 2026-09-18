# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU Chronos-2 single forward, or Chronos-1 tokenizer smoke demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.experimental.chronos_forecast.common.configs import ChronosModelConfig
from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model
from models.experimental.chronos_forecast.reference.pytorch_chronos import create_chronos_config

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "weights" / "chronos-2"


def run_chronos2_forward(checkpoint: Path, context_length: int) -> None:
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"No Chronos-2 checkpoint at {checkpoint}. Download with:\n"
            "  hf download amazon/chronos-2 "
            "--local-dir models/experimental/chronos_forecast/weights/chronos-2"
        )
    model = Chronos2Model.from_pretrained(checkpoint).eval()
    context = torch.sin(torch.linspace(0, 4 * torch.pi, context_length)).unsqueeze(0)
    with torch.no_grad():
        out = model(context=context, num_output_patches=1)
    preds = out.quantile_preds
    print("Chronos-2 single forward")
    print(f"  checkpoint:         {checkpoint}")
    print(f"  context shape:      {tuple(context.shape)}")
    print(f"  quantile_preds:     {tuple(preds.shape)}")
    print(f"  median[0, :8]:      {preds[0, preds.shape[1] // 2, :8].tolist()}")


def run_tokenizer_demo(context_length: int) -> None:
    config = create_chronos_config(ChronosModelConfig())
    tokenizer = config.create_tokenizer()
    context = torch.sin(torch.linspace(0, 4 * torch.pi, context_length)).unsqueeze(0)
    token_ids, attention_mask, _ = tokenizer.context_input_transform(context)
    print("Chronos-1 tokenizer demo")
    print(f"  context shape:      {tuple(context.shape)}")
    print(f"  token_ids shape:    {tuple(token_ids.shape)}")
    print(f"  attention_mask sum: {int(attention_mask.sum().item())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos reference demo")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help="Local amazon/chronos-2 snapshot for a Chronos2Model.forward",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Run the Chronos-1 tokenizer smoke path instead of Chronos-2 forward",
    )
    args = parser.parse_args()
    if args.tokenizer_only:
        run_tokenizer_demo(args.context_length)
        return
    run_chronos2_forward(args.checkpoint, args.context_length)


if __name__ == "__main__":
    main()
