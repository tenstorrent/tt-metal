# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


def memory(kind, start, size, shard, col=False):
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(start[0] + size[0] - 1, start[1] + size[1] - 1))}
    )
    return ttnn.MemoryConfig(
        getattr(ttnn.TensorMemoryLayout, kind),
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR if col else ttnn.ShardOrientation.ROW_MAJOR),
    )


def put(device, x, mem=ttnn.DRAM_MEMORY_CONFIG, rm=False):
    return ttnn.from_torch(
        x,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )


def compare(actual, expected):
    got = (actual if isinstance(actual, torch.Tensor) else ttnn.to_torch(actual)).float()
    expected = expected.float()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    wrong = got != expected
    return dict(wrong=int(wrong.sum()), elements=got.numel(), max_abs=float((got - expected).abs().max()))


def values(shape, op, start, high=100, salt=0):
    # Distinct placement data prevents stale L1 contents from hiding misplaced reads.
    seed = 100000 * start[0] + 10000 * start[1] + 100 * salt + sum((i + 1) * ord(c) for i, c in enumerate(op))
    return torch.randint(-high, high + 1, shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def untilize_case(device, op, start, col):
    # quasar_untilize keeps part of every shard; drop_rows and drop_cols discard the second row or column of shards
    # whole.
    end = {"quasar_untilize": (47, 47), "quasar_untilize_drop_rows": (31, 47), "quasar_untilize_drop_cols": (47, 31)}[
        op
    ]
    x = values((1, 1, 64, 64), op, start, salt=int(col))
    a = put(device, x, memory("BLOCK_SHARDED", start, (2, 2), (32, 32), col=col))
    y = ttnn.experimental.quasar.untilize_with_unpadding(a, (0, 0, *end), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return [compare(y, x[..., : end[0] + 1, : end[1] + 1])]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [(op, origin, col) for op in ["quasar_untilize"] for origin in FOUR_ORIGINS for col in [False, True]] + [
    (op, origin, False)
    for op in ["quasar_untilize_drop_rows", "quasar_untilize_drop_cols"]
    for origin in [(0, 0), (2, 3)]
]


@pytest.mark.parametrize("op,origin,col", CASES)
def test_quasar_untilize_with_unpadding_shard_grid_origin(device, op, origin, col):
    grid = device.compute_with_storage_grid_size()
    if origin[0] + 2 > grid.x or origin[1] + 2 > grid.y:
        pytest.skip(f"needs a {origin[0] + 2}x{origin[1] + 2} worker grid; the device has {grid.x}x{grid.y}")
    outputs = untilize_case(device, op, origin, col)
    assert all(output["wrong"] == 0 for output in outputs), outputs
