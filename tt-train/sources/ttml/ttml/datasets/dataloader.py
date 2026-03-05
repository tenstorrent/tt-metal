# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Batch dataclass and TTMLDataloader abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


@dataclass
class Batch:
    """A single training batch in ttml-native tensor format.

    Shapes follow ttml conventions:
        input_ids:  [B, 1, 1, T]  uint32  — token IDs fed into the model
        labels:     [B, T]        uint32  — next-token targets (shifted from input_ids)
        loss_mask:  [B, 1, T, 1]  bfloat16 — 0.0 for prompt/padding positions,
                                              nonzero (normalized) for completion positions
    """

    input_ids: Any
    labels: Any
    loss_mask: Any


class TTMLDataloader(ABC):
    """Abstract dataloader that yields :class:`Batch` objects.

    Concrete subclasses own the iteration strategy (sequential, shuffled,
    streaming, etc.).  The collate function is injected rather than hardcoded
    so the same dataloader class can be reused across different algorithms by
    swapping only the collate logic.

    Example::

        from functools import partial
        from ttml.datasets import TTMLDataloader, Batch

        class SimpleDataloader(TTMLDataloader):
            def __iter__(self):
                for i in range(0, len(self.dataset) - self.batch_size, self.batch_size):
                    examples = [self.dataset[j] for j in range(i, i + self.batch_size)]
                    yield self.collate_fn(examples)

            def __len__(self):
                return len(self.dataset) // self.batch_size
    """

    def __init__(
        self,
        dataset: Any,
        collate_fn: Callable,
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self) -> Iterator[Batch]:
        """Yield batches until the dataset is exhausted."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of batches available in one pass over the dataset."""
        ...
