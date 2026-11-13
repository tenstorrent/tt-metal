# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Datasets by ``dataset.id``, and the dataloaders over them."""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

from ..configs import SFTTrainerConfig
from .alpaca import AlpacaDataset
from .collate import collate_causal_lm
from .custom import CustomDataset
from .llm_dataset import IGNORED_INDEX, LLMDataset
from .sst2 import SST2Dataset

DATASETS: dict[str, type[LLMDataset]] = {"sst2": SST2Dataset, "alpaca": AlpacaDataset, "custom": CustomDataset}


def get_dataset(config: SFTTrainerConfig, split: str, tokenizer: Any) -> LLMDataset:
    dataset_id = config.dataset.id.lower()
    if dataset_id not in DATASETS:
        raise ValueError(f"dataset.id: {config.dataset.id!r} is not one of {sorted(DATASETS)}")
    return DATASETS[dataset_id](config, split, tokenizer)


def load_tokenizer(model_name: str) -> Any:
    """The model's HF tokenizer, padding on the right; EOS pads when the model has no pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(dataset: LLMDataset, config: SFTTrainerConfig, shuffle: bool) -> Optional[Any]:
    """A ttml dataloader over ``dataset`` yielding host batches (``collate_causal_lm``); None for an unconfigured
    split. Full batches only."""
    if dataset.dataset is None:
        return None
    from ttml.datasets import InMemoryDataloader

    collate = partial(collate_causal_lm, max_length=config.max_length, pad_token_id=dataset.tokenizer.pad_token_id)
    return InMemoryDataloader(
        dataset, collate, batch_size=config.batch_size, shuffle=shuffle, drop_last=True, seed=config.seed
    )


def clear_cache() -> None:
    LLMDataset.clear_cache()


__all__ = [
    "AlpacaDataset",
    "CustomDataset",
    "DATASETS",
    "IGNORED_INDEX",
    "LLMDataset",
    "SST2Dataset",
    "build_dataloader",
    "clear_cache",
    "collate_causal_lm",
    "get_dataset",
    "load_tokenizer",
]
