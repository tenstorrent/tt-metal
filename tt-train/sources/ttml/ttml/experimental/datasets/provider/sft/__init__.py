# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SFT dataset providers: the batches a supervised fine-tuning run trains and validates on, by ``dataset.id``.

A provider serves one split of a dataset as host batches, which the trainer asks it for one at a time; the trainer
never sees examples, prompts or tokenization. Adding a dataset means subclassing ``DatasetProvider`` (``base.py``) in
``ttml/experimental/datasets/provider/sft/<dataset>.py``, with its prompts in
``ttml/experimental/datasets/utils/sft/templates/<dataset>.py``, and adding it to ``providers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from ...utils.sft.collate import IGNORED_INDEX
from .alpaca import AlpacaProvider
from .base import DatasetProvider
from .custom import CustomProvider
from .sst2 import SST2Provider

if TYPE_CHECKING:
    from ttml.experimental.trainer.configs import SFTTrainerConfig


def providers() -> dict[str, type[DatasetProvider]]:
    """The provider of each dataset, keyed by its ``dataset.id``."""
    return {"sst2": SST2Provider, "alpaca": AlpacaProvider, "custom": CustomProvider}


def get_providers(
    config: SFTTrainerConfig, tokenizer: Any, splits: Sequence[str]
) -> dict[str, Optional[DatasetProvider]]:
    """The provider of each of ``splits`` of the dataset ``config.dataset.id`` names; None for a split the dataset
    doesn't have (a custom dataset without validation files). A dataset whose splits are carved out of one source is
    loaded and tokenized once, for all of them."""
    provider = providers().get(config.dataset.id.lower())
    if provider is None:
        raise ValueError(f"dataset.id: {config.dataset.id!r} is not one of {sorted(providers())}")
    built: dict[str, Optional[DatasetProvider]] = {}
    source = None
    for split in splits:
        if not provider.has_split(config, split):
            built[split] = None
            continue
        built[split] = provider(config, split, tokenizer, source)
        source = built[split].source
    return built


def load_tokenizer(model_name: str) -> Any:
    """The HF tokenizer of the model at ``model_name``, a hub id or a local directory, padding on the right; EOS pads
    when the model has no pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


__all__ = [
    "IGNORED_INDEX",
    "AlpacaProvider",
    "CustomProvider",
    "DatasetProvider",
    "SST2Provider",
    "get_providers",
    "load_tokenizer",
    "providers",
]
