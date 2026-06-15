# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.sampling.vocab_padding import build_invalid_vocab_mask, get_vocab_shard_dims


def test_invalid_vocab_mask_not_needed_without_padding():
    assert build_invalid_vocab_mask(vocab_size=128256, padded_vocab_size=128256, max_batch_size=32) is None


def test_invalid_vocab_mask_marks_only_padded_tokens():
    mask = build_invalid_vocab_mask(vocab_size=10, padded_vocab_size=16, max_batch_size=2)

    assert mask.shape == (1, 1, 2, 16)
    assert mask.dtype == torch.bfloat16
    assert torch.all(mask[..., :10] == 0)
    assert torch.all(mask[..., 10:] == torch.finfo(torch.bfloat16).min)


@pytest.mark.parametrize(
    "cluster_shape,sampling_all_gather_axis,expected",
    [
        ((1, 1), 0, (None, None)),
        ((1, 8), 0, (None, 3)),
        ((8, 1), 0, (3, None)),
        ((8, 4), 0, (3, None)),
        ((8, 4), 1, (None, 3)),
    ],
)
def test_vocab_shard_dims_follow_sampling_tp_axis(cluster_shape, sampling_all_gather_axis, expected):
    assert get_vocab_shard_dims(cluster_shape, sampling_all_gather_axis) == expected
