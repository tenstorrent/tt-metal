# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side batching. NumPy only."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

# The label of a position that isn't trained on, as in HF: the prompt and the padding.
IGNORED_INDEX = -100


def collate_causal_lm(
    examples: Sequence[Mapping], max_length: int, pad_token_id: int, batch_size: Optional[int] = None
) -> dict[str, np.ndarray]:
    """Pad tokenized examples to ``max_length`` and build next-token targets in ttml's batch layout.

    ``B`` is ``batch_size``, or ``len(examples)`` when it is None. The rows past the examples are padding rows: all
    ``pad_token_id``, with no targets: a short last batch keeps the shape of the others, and its loss is its examples'.

    - ``input_ids``  ``[B, 1, 1, T]`` uint32, right-padded with ``pad_token_id``;
    - ``labels``     ``[B, T]`` uint32, the next token at each position (ignored positions hold
      ``pad_token_id``, since uint32 can't hold ``-100``);
    - ``loss_mask``  ``[B, 1, T, 1]`` float32, 1 at positions with a real target, scaled to sum to ``B * T``:
      ``mean(cross_entropy * loss_mask)`` is then the mean over real targets.

    The same loss as HF's on one batch: right padding with ``-100`` labels, labels shifted by one, and
    ``cross_entropy(..., ignore_index=-100)``. With gradient accumulation the trainers average the batches' losses
    equally, as ``ttml.trainers.SFTTrainer`` does, where HF's Trainer weights each by its share of the window's
    targets; the two match only with ``gradient_accumulation_steps=1``.
    """
    if batch_size is None:
        batch_size = len(examples)
    elif len(examples) > batch_size:
        raise ValueError(f"{len(examples)} examples don't fit in a batch of {batch_size}")
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
