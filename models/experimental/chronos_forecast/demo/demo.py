# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU demo: tokenize a synthetic series with the Amazon Chronos reference."""

from __future__ import annotations

import argparse

import torch

from models.experimental.chronos_forecast.common.configs import ChronosModelConfig
from models.experimental.chronos_forecast.reference.pytorch_chronos import create_chronos_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos reference tokenizer smoke demo")
    parser.add_argument("--context-length", type=int, default=16)
    args = parser.parse_args()

    config = create_chronos_config(ChronosModelConfig())
    tokenizer = config.create_tokenizer()
    context = torch.sin(torch.linspace(0, 4 * torch.pi, args.context_length)).unsqueeze(0)
    token_ids, attention_mask, _ = tokenizer.context_input_transform(context)

    print("Chronos reference tokenizer demo")
    print(f"  context shape:      {tuple(context.shape)}")
    print(f"  token_ids shape:    {tuple(token_ids.shape)}")
    print(f"  attention_mask sum: {int(attention_mask.sum().item())}")
    print("TTNN forward is not implemented yet.")


if __name__ == "__main__":
    main()
