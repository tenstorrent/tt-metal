# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for ``ttml.fsdp``'s automatic shard-dim selection.

The CCL ops behind FSDP fall onto a slow composite path whenever a per-rank shard is
not tile-aligned (a multiple of 32 on the gathered dim), so ``"auto"`` must prefer an
aligned dim whenever one exists while keeping the historical ``rank-2`` preference
otherwise.
"""

import pytest

from ttml.fsdp import _pick_shard_dim_from_shape, _shard_is_tile_aligned


@pytest.mark.parametrize(
    "shape, axis_size, expected",
    [
        # Both dims aligned -> keep the historical rows-first preference.
        ((1, 1, 2048, 2048), 8, 2),
        ((1, 1, 5632, 2048), 8, 2),
        # Rows misaligned at N=32 (176-row shards), cols aligned -> cols.
        ((1, 1, 5632, 2048), 32, 3),
        # kv projection: 16-row shards at N=32 -> cols.
        ((1, 1, 512, 2048), 32, 3),
        # Embedding / LM head: 1000-row shards at N=32 -> cols.
        ((1, 1, 32000, 2048), 32, 3),
        # Llama-3 vocab at N=32: 4008-row shards misaligned -> cols (4096/32 = 128).
        ((1, 1, 128256, 4096), 32, 3),
        # Neither dim aligned -> fall back to rows (caller warns).
        ((1, 1, 5632, 176 * 32), 32, 2),
        # Rows not divisible by N at all -> cols.
        ((1, 1, 100, 2048), 8, 3),
        # Norm gamma: dim 2 has size 1 -> cols.
        ((1, 1, 1, 2048), 8, 3),
        # Nothing divisible -> None.
        ((1, 1, 100, 100), 8, None),
    ],
)
def test_auto_shard_dim_prefers_tile_aligned(shape, axis_size, expected):
    assert _pick_shard_dim_from_shape(list(shape), set(), 0, axis_size) == expected


def test_auto_shard_dim_skips_dims_taken_by_other_axes():
    # TP already shards rows (ColumnParallelLinear) -> FSDP must take cols even though rows are aligned.
    assert _pick_shard_dim_from_shape([1, 1, 1024, 4096], {2}, 0, 8) == 3
    # TP shards cols (RowParallelLinear) -> FSDP takes rows.
    assert _pick_shard_dim_from_shape([1, 1, 4096, 1024], {3}, 0, 8) == 2


def test_auto_shard_dim_without_axis_size_keeps_legacy_behaviour():
    # No axis_size: no divisibility / alignment filtering, rows first.
    assert _pick_shard_dim_from_shape([1, 1, 5632, 2048], set(), 0) == 2
    assert _pick_shard_dim_from_shape([1, 1, 1, 2048], set(), 0) == 3


def test_shard_is_tile_aligned():
    assert _shard_is_tile_aligned(2048, 8)
    assert _shard_is_tile_aligned(2048, 32)
    assert not _shard_is_tile_aligned(5632, 32)
    assert not _shard_is_tile_aligned(512, 32)
    assert not _shard_is_tile_aligned(100, 8)
