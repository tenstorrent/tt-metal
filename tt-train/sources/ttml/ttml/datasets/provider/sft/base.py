# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np

from ...hf_dataloader import InMemoryDataloader
from ...utils.sft.collate import IGNORED_INDEX, collate_causal_lm

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import SFTTrainerConfig


class DatasetProvider:
    """One split of a dataset for causal-LM fine-tuning: prompt/response examples, tokenized and served as batches.

    The trainer takes an epoch's batches one at a time, or by iterating the provider; ``len`` is the batches of
    an epoch::

        train = get_provider(config, "train", tokenizer)
        while (batch := train.next_batch()) is not None:
            ...

    ``stream`` runs the epochs one after another, as a training loop takes them.

    A batch is ``config.batch_size`` examples as host arrays (``collate_causal_lm``); a last partial batch is dropped.
    The train split is shuffled, in an order that depends only on the seed and the epoch, which ``seek`` sets.

    Each example is ``input_ids`` (prompt, response and EOS) and ``labels``, which are ``input_ids`` with the
    prompt masked to ``IGNORED_INDEX``: only the response is trained on. Examples longer than ``max_length``
    are dropped.

    Subclasses provide the raw HF dataset (``_load_raw_dataset``) and how one example maps to a prompt and
    a response (``_prompt_and_response``). A source that ships one split carves validation out of it
    (``train_val_split_ratio``), and one that may lack a split says so (``has_split``).

    Examples are lists of token ids: the provider needs no device, and the tokenizer is passed in.
    """

    # Sources with only a train split carve validation out of it with this ratio; None means the source
    # has the requested split natively.
    train_val_split_ratio: Optional[float] = None

    def __init__(self, config: SFTTrainerConfig, split: str, tokenizer: Any) -> None:
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        # A `datasets.Dataset` of the split's tokenized examples.
        self.examples: Any = self._prepare_examples()
        collate = partial(collate_causal_lm, max_length=config.max_length, pad_token_id=tokenizer.pad_token_id)
        self._dataloader = InMemoryDataloader(
            self.examples,
            collate,
            batch_size=config.batch_size,
            shuffle=split == "train",
            drop_last=True,
            seed=config.seed,
        )
        # The batches left in the current epoch; None between epochs.
        self._epoch_batches: Optional[Iterator[dict[str, np.ndarray]]] = None

    # ------------------------------------------------------------------ for subclasses

    @classmethod
    def has_split(cls, config: SFTTrainerConfig, split: str) -> bool:
        """Whether the dataset ``config`` names has ``split`` at all."""
        return True

    def _load_raw_dataset(self) -> Any:
        """The raw HF dataset of ``self.split`` (or the whole source, with ``train_val_split_ratio``)."""
        raise NotImplementedError

    def _prompt_and_response(self, example: dict) -> tuple[str, str]:
        raise NotImplementedError

    # ------------------------------------------------------------------ batches

    def __len__(self) -> int:
        """The batches of an epoch."""
        return len(self._dataloader)

    @property
    def num_examples(self) -> int:
        return len(self.examples)

    def next_batch(self) -> Optional[dict[str, np.ndarray]]:
        """The next batch of the epoch; None once the epoch is over, after which the next epoch starts."""
        if self._epoch_batches is None:
            self._epoch_batches = iter(self._dataloader)
        batch = next(self._epoch_batches, None)
        if batch is None:
            self._epoch_batches = None
        return batch

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """The batches left in the epoch."""
        while (batch := self.next_batch()) is not None:
            yield batch

    def seek(self, epoch: int) -> None:
        """Continue with the first batch of epoch ``epoch``."""
        self._dataloader.set_state_dict({"seed": self.config.seed, "epoch": epoch, "position": 0})
        self._epoch_batches = None

    def stream(self, start_epoch: int = 0) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """``(epoch, batch)`` for every batch of epoch ``start_epoch`` and of each epoch after it, without end; each
        epoch in the order ``seek`` sets for it."""
        for epoch in itertools.count(start_epoch):
            self.seek(epoch)
            for batch in self:
                yield epoch, batch

    # ------------------------------------------------------------------ examples

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

    def _prepare_examples(self) -> Any:
        if self.train_val_split_ratio is None:
            return self._tokenize_and_filter(self._load_raw_dataset())

        full = self._tokenize_and_filter(self._load_raw_dataset()).shuffle(seed=self.config.seed)
        train_end = int(self.train_val_split_ratio * len(full))
        if self.split == "train":
            return full.select(range(0, train_end))
        if self.split == "validation":
            return full.select(range(train_end, len(full)))
        raise ValueError(f"{type(self).__name__} has the splits 'train' and 'validation', not {self.split!r}")
