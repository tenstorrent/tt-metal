# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete TTMLDataloader and SFT collate function for HuggingFace datasets."""

from typing import Any, Callable, Iterator

import numpy as np
import ttnn

import ttml
from ttml.datasets.dataloader import Batch, TTMLDataloader


class InMemoryDataloader(TTMLDataloader):
    """Concrete :class:`TTMLDataloader` for in-memory and HuggingFace datasets.

    Iterates the dataset by index, groups examples into batches, and calls the
    injected ``collate_fn`` to produce :class:`Batch` objects with ttml tensors.

    Yields examples indefinitely — when the dataset is exhausted it reshuffles
    (if ``shuffle=True``) and starts again, so the training loop never sees a
    ``StopIteration``.

    Example::

        from functools import partial
        from datasets import load_dataset
        from ttml.datasets import InMemoryDataloader, sft_collate_fn

        dataset = load_dataset("trl-lib/Capybara", split="train")
        collate = partial(sft_collate_fn, max_seq_len=2048, pad_token_id=tokenizer.pad_token_id)
        loader = InMemoryDataloader(dataset, collate, batch_size=8, shuffle=True)
    """

    def __init__(
        self,
        dataset: Any,
        collate_fn: Callable,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        super().__init__(dataset, collate_fn, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Batch]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            np.random.shuffle(indices)
        end = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, end, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            examples = [self.dataset[j] for j in batch_indices]
            yield self.collate_fn(examples)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


def sft_collate_fn(
    examples: list,
    max_seq_len: int,
    pad_token_id: int,
) -> Batch:
    """Collate a list of SFT examples into a :class:`Batch` with ttml tensors.

    Each example must be a dict with:

    * ``"input_ids"`` — list of token IDs for the full sequence
      (prompt + completion concatenated).
    * ``"labels"`` — list of token IDs the same length as ``input_ids``,
      where prompt positions are set to ``-100`` (loss is not computed there).

    The function:

    1. Pads sequences to the longest in the batch (capped at ``max_seq_len``).
    2. Converts ``labels == -100`` positions to ``pad_token_id`` so ttml's
       uint32 label tensors stay valid, and records which positions are masked.
    3. Normalises ``loss_mask`` so that ``mean(loss * loss_mask)`` equals the
       per-completion-token loss regardless of batch composition.

    Returns a :class:`Batch` with ttml tensors in the standard shapes:

    * ``input_ids``  — ``[B, 1, 1, T]`` uint32
    * ``labels``     — ``[B, T]``        uint32
    * ``loss_mask``  — ``[B, 1, T, 1]`` bfloat16
    """
    batch_size = len(examples)
    seq_len = min(
        max(len(e["input_ids"]) for e in examples),
        max_seq_len,
    )

    input_ids_np = np.full((batch_size, seq_len), pad_token_id, dtype=np.uint32)
    labels_np = np.full((batch_size, seq_len), pad_token_id, dtype=np.uint32)
    loss_mask_np = np.zeros((batch_size, 1, seq_len, 1), dtype=np.float32)

    for i, ex in enumerate(examples):
        ids = list(ex["input_ids"])[:seq_len]
        lbs = list(ex["labels"])[:seq_len]
        n = len(ids)
        input_ids_np[i, :n] = ids
        for t, lb in enumerate(lbs):
            if lb != -100:
                labels_np[i, t] = lb
                loss_mask_np[i, 0, t, 0] = 1.0
            else:
                labels_np[i, t] = pad_token_id

    # Normalise: scale so mean(loss * loss_mask) = per-completion-token loss.
    total = loss_mask_np.sum()
    if total > 0:
        loss_mask_np *= (batch_size * seq_len) / total

    return Batch(
        input_ids=ttml.autograd.Tensor.from_numpy(
            input_ids_np.reshape(batch_size, 1, 1, seq_len),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
        ),
        labels=ttml.autograd.Tensor.from_numpy(
            labels_np,
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
        ),
        loss_mask=ttml.autograd.Tensor.from_numpy(
            loss_mask_np,
            ttnn.Layout.TILE,
            ttnn.DataType.BFLOAT16,
        ),
    )
