# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .provider import DatasetProvider
from .templates import ALPACA_PROMPT_TEMPLATE, ALPACA_PROMPT_TEMPLATE_NO_INPUT


class AlpacaProvider(DatasetProvider):
    """tatsu-lab/alpaca instruction following."""

    # Alpaca only has a train split; validation is carved out of it.
    train_val_split_ratio = 0.98

    def _load_raw_dataset(self) -> Any:
        from datasets import load_dataset

        return load_dataset("tatsu-lab/alpaca", split="train")

    def _prompt_and_response(self, example: dict) -> tuple[str, str]:
        instruction = example["instruction"]
        input_text = example.get("input", "")
        if input_text.strip():
            prompt = ALPACA_PROMPT_TEMPLATE.substitute(instruction=instruction, input=input_text)
        else:
            prompt = ALPACA_PROMPT_TEMPLATE_NO_INPUT.substitute(instruction=instruction)
        return prompt, example["output"]
