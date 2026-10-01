# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""FSDP shard-dim selection and tile alignment (shape-only, no device needed)."""

from __future__ import annotations

import pytest

from ttml.fsdp import _is_tile_aligned_shard, _pick_shard_dim_from_shape


@pytest.mark.parametrize(
    "shape, axis_size, expected",
    [
        # rank-2 already splits into whole tiles: keep the default choice.
        ([1, 1, 1024, 384], 8, 2),
        # rank-2 would be 48-row shards, rank-1 gives 128-column shards.
        ([1, 1, 384, 1024], 8, 3),
        # Llama-3 vocab embedding over 32 devices: 4008 rows is not whole tiles, 128 columns is.
        ([1, 1, 128256, 4096], 32, 3),
        # Neither dim aligns: keep the default order.
        ([1, 1, 384, 384], 8, 2),
        # rank-2 doesn't even divide, rank-1 does (misaligned): prefer the divisible dim.
        ([1, 1, 100, 48], 8, 3),
        # Norm weight: rank-2 has size 1, so rank-1 is the only candidate.
        ([1, 1, 1, 384], 8, 3),
    ],
)
def test_pick_shard_dim_prefers_tile_aligned(shape, axis_size, expected):
    assert _pick_shard_dim_from_shape(shape, set(), axis_index=1, axis_size=axis_size) == expected


def test_pick_shard_dim_skips_dims_sharded_by_another_axis():
    # rank-1 is TP-sharded, so rank-2 is used even though its shards are misaligned.
    assert _pick_shard_dim_from_shape([1, 1, 384, 1024], {3}, axis_index=1, axis_size=8) == 2


def test_pick_shard_dim_without_axis_size_keeps_legacy_order():
    assert _pick_shard_dim_from_shape([1, 1, 384, 1024], set(), axis_index=1) == 2


def test_pick_shard_dim_with_no_candidate_returns_none():
    assert _pick_shard_dim_from_shape([1, 1, 1, 1], set(), axis_index=1, axis_size=8) is None


@pytest.mark.parametrize(
    "shape, dim, axis_size, aligned",
    [
        ([1, 1, 512, 384], 2, 8, True),  # 64 rows
        ([1, 1, 384, 384], 2, 8, False),  # 48 rows
        ([1, 1, 1, 128], 3, 8, False),  # Qwen3 q/k norm: 16 columns
        ([1, 1, 1, 128], 3, 4, True),  # 32 columns
        ([1, 1, 100, 64], 2, 8, False),  # not even divisible
        ([8, 1, 48, 48], 0, 8, True),  # outer (non-tiled) dim: any even split is fine
    ],
)
def test_is_tile_aligned_shard(shape, dim, axis_size, aligned):
    assert _is_tile_aligned_shard(shape, dim, axis_size) is aligned
