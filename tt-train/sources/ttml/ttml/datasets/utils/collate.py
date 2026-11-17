# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side batching. NumPy only."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# The label of a position that isn't trained on, as in HF: the prompt and the padding.
IGNORED_INDEX = -100


def collate_causal_lm(examples: Sequence[Mapping], max_length: int, pad_token_id: int) -> dict[str, np.ndarray]:
    """Pad tokenized examples to ``max_length`` and build next-token targets in ttml's batch layout.

    - ``input_ids``  ``[B, 1, 1, T]`` uint32, right-padded with ``pad_token_id``;
    - ``labels``     ``[B, T]`` uint32, the next token at each position (ignored positions hold
      ``pad_token_id``, since uint32 can't hold ``-100``);
    - ``loss_mask``  ``[B, 1, T, 1]`` float32, 1 at positions with a real target, scaled to sum to ``B * T``:
      ``mean(cross_entropy * loss_mask)`` is then the mean over real targets.

    The same loss as HF's: right padding with ``-100`` labels, labels shifted by one, and
    ``cross_entropy(..., ignore_index=-100)``.
    """
    batch_size = len(examples)
    input_ids = np.full((batch_size, max_length), pad_token_id, dtype=np.uint32)
    labels = np.full((batch_size, max_length), IGNORED_INDEX, dtype=np.int64)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        if length > max_length:
            raise ValueError(f"Example of {length} tokens is longer than max_length={max_length}")
        input_ids[row, :length] = example["input_ids"]
        labels[row, :length] = example["labels"]

    # Position t predicts token t + 1; the last position has no target.
    targets = np.full_like(labels, IGNORED_INDEX)
    targets[:, :-1] = labels[:, 1:]
    valid = targets != IGNORED_INDEX

    loss_mask = valid.astype(np.float32)
    num_valid = loss_mask.sum()
    if num_valid > 0:
        loss_mask *= loss_mask.size / num_valid
    targets[~valid] = pad_token_id

    return {
        "input_ids": input_ids.reshape(batch_size, 1, 1, max_length),
        "labels": targets.astype(np.uint32),
        "loss_mask": loss_mask.reshape(batch_size, 1, max_length, 1),
    }
