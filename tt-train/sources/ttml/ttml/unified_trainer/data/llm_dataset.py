# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..configs import SFTTrainerConfig

IGNORED_INDEX = -100


class LLMDataset:
    """Prompt/response examples for causal-LM fine-tuning, tokenized.

    Each example is ``input_ids`` (prompt, response and EOS) and ``labels``, which are ``input_ids`` with the
    prompt masked to ``IGNORED_INDEX``: only the response is trained on. Examples longer than ``max_length``
    are dropped.

    Subclasses provide the raw HF dataset (``_load_raw_dataset``) and how one example maps to a prompt and
    a response (``_prompt_and_response``). A source that ships one split carves validation out of it
    (``train_val_split_ratio``).

    Examples are lists of token ids: the dataset needs no framework, and the tokenizer is passed in.
    """

    # Sources with only a train split carve validation out of it with this ratio; None means the source
    # has the requested split natively.
    train_val_split_ratio: Optional[float] = None
    # Tokenized single-split sources, so one is tokenized once for both splits of a run. Keyed by all the
    # tokens depend on; the run's owner clears it (`clear_cache`).
    _shared_datasets: ClassVar[dict[tuple, Any]] = {}

    def __init__(self, config: SFTTrainerConfig, split: str, tokenizer: Any) -> None:
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        # A `datasets.Dataset`, or None when the split isn't configured (a custom dataset without validation).
        self.dataset: Any = None
        self._prepare_dataset()

    def _load_raw_dataset(self) -> Any:
        """The raw HF dataset of ``self.split`` (or the whole source, with ``train_val_split_ratio``); None if
        the split isn't configured."""
        raise NotImplementedError

    def _prompt_and_response(self, example: dict) -> tuple[str, str]:
        raise NotImplementedError

    def _tokenize(self, example: dict) -> dict:
        prompt, response = self._prompt_and_response(example)
        input_ids = self.tokenizer(prompt + response + self.tokenizer.eos_token, truncation=False, padding=False)[
            "input_ids"
        ]
        prompt_len = len(self.tokenizer(prompt, truncation=False, padding=False)["input_ids"])
        return {
            "input_ids": input_ids,
            "labels": [IGNORED_INDEX if i < prompt_len else token for i, token in enumerate(input_ids)],
        }

    def _tokenize_and_filter(self, raw_dataset: Any) -> Any:
        tokenized = raw_dataset.map(self._tokenize, remove_columns=raw_dataset.column_names)
        max_length = self.config.max_length
        return tokenized.filter(lambda example: len(example["input_ids"]) <= max_length)

    def _prepare_dataset(self) -> None:
        if self.train_val_split_ratio is None:
            raw_dataset = self._load_raw_dataset()
            self.dataset = None if raw_dataset is None else self._tokenize_and_filter(raw_dataset)
            return

        key = self._cache_key()
        if key not in LLMDataset._shared_datasets:
            full = self._tokenize_and_filter(self._load_raw_dataset())
            LLMDataset._shared_datasets[key] = full.shuffle(seed=self.config.seed)
        full = LLMDataset._shared_datasets[key]
        train_end = int(self.train_val_split_ratio * len(full))
        if self.split == "train":
            self.dataset = full.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full.select(range(train_end, len(full)))
        else:
            raise ValueError(f"{type(self).__name__} has the splits 'train' and 'validation', not {self.split!r}")

    def _cache_key(self) -> tuple:
        return (type(self), self.config.model.name, self.config.max_length, self.config.seed)

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the tokenized sources cached for this run (call when the run is cleaned up)."""
        LLMDataset._shared_datasets.clear()

    def __len__(self) -> int:
        return 0 if self.dataset is None else len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        return {"input_ids": sample["input_ids"], "labels": sample["labels"]}
