# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.sampling.vocab_padding import (
    build_invalid_vocab_mask,
    build_tail_invalid_vocab_mask,
    get_vocab_num_shards,
    get_vocab_shard_dims,
)


def test_invalid_vocab_mask_not_needed_without_padding():
    assert build_invalid_vocab_mask(vocab_size=128256, padded_vocab_size=128256, max_batch_size=32) is None


def test_invalid_vocab_mask_marks_only_padded_tokens():
    mask = build_invalid_vocab_mask(vocab_size=10, padded_vocab_size=16, max_batch_size=2)

    assert mask.shape == (1, 1, 2, 16)
    assert mask.dtype == torch.bfloat16
    assert torch.all(mask[..., :10] == 0)
    assert torch.all(mask[..., 10:] == torch.finfo(torch.bfloat16).min)


def test_invalid_vocab_mask_prevents_padded_token_argmax():
    logits = torch.full((1, 1, 1, 16), -1.0, dtype=torch.bfloat16)
    logits[..., 10:] = 0.0

    assert logits.argmax(dim=-1).item() == 10

    mask = build_invalid_vocab_mask(vocab_size=10, padded_vocab_size=16, max_batch_size=1)
    masked_logits = logits + mask

    assert masked_logits.argmax(dim=-1).item() == 0


def test_tail_invalid_vocab_mask_not_needed_without_padding():
    assert (
        build_tail_invalid_vocab_mask(
            vocab_size=128256,
            padded_vocab_size=128256,
            max_batch_size=32,
            cluster_shape=(1, 8),
        )
        is None
    )


def test_tail_invalid_vocab_mask_matches_qwen3_32b_t3k_layout():
    tail_mask = build_tail_invalid_vocab_mask(
        vocab_size=151936,
        padded_vocab_size=152064,
        max_batch_size=32,
        cluster_shape=(1, 8),
    )

    assert tail_mask is not None
    assert tail_mask.tail_width == 128
    assert tail_mask.shard_width == 19008
    assert tail_mask.num_vocab_shards == 8
    assert tail_mask.mask.shape == (1, 1, 32, 1024)

    min_value = torch.finfo(torch.bfloat16).min
    for shard_id in range(7):
        shard_slice = tail_mask.mask[..., shard_id * 128 : (shard_id + 1) * 128]
        assert torch.all(shard_slice == 0)
    assert torch.all(tail_mask.mask[..., 7 * 128 :] == min_value)


def test_tail_invalid_vocab_mask_accepts_ttnn_mesh_shape_like_iterable():
    class MeshShapeLike:
        def __iter__(self):
            return iter((1, 8))

    tail_mask = build_tail_invalid_vocab_mask(
        vocab_size=151936,
        padded_vocab_size=152064,
        max_batch_size=32,
        cluster_shape=MeshShapeLike(),
    )

    assert tail_mask is not None
    assert tail_mask.num_vocab_shards == 8


def test_tail_invalid_vocab_mask_falls_back_for_non_tile_aligned_tail():
    assert (
        build_tail_invalid_vocab_mask(
            vocab_size=10,
            padded_vocab_size=16,
            max_batch_size=2,
            cluster_shape=(1, 1),
        )
        is None
    )


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


@pytest.mark.parametrize(
    "cluster_shape,sampling_all_gather_axis,expected",
    [
        ((1, 1), 0, 1),
        ((1, 8), 0, 8),
        ((8, 1), 0, 8),
        ((8, 4), 0, 8),
        ((8, 4), 1, 4),
    ],
)
def test_vocab_num_shards_follow_sampling_tp_axis(cluster_shape, sampling_all_gather_axis, expected):
    assert get_vocab_num_shards(cluster_shape, sampling_all_gather_axis) == expected
