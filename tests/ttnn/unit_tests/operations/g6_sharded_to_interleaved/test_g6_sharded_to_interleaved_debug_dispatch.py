# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug: the program path on cells outside Phase 0 SUPPORTED, bypassing the support gate only.

The kernel and page map are the same for every TARGET value (design: TARGET axes as placement). These
cases drive the structural checks + dispatch directly so a later refinement that widens SUPPORTED
starts from a verified path. Exact bit compare; per-tile scaled input so a wrong page map shows.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import role_dump
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _check_structure, _dispatch

TILE = 32

P2_TO_L1 = dict(
    shape=(1, 5120),
    dtype=ttnn.bfloat16,
    scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(32, 160),
    grid=(((0, 0), (7, 3)),),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.L1_MEMORY_CONFIG,
)
P1_PANOPTIC = dict(
    shape=(1, 128, 256, 32),
    dtype=ttnn.bfloat8_b,
    scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(320, 32),
    grid=(((0, 0), (10, 8)), ((0, 9), (3, 9))),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.L1_MEMORY_CONFIG,
)
P1_OTHER_SPEC = {
    **P1_PANOPTIC,
    "shard": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": ttnn.ShardOrientation.COL_MAJOR,
}
P1_TO_DRAM = {**P1_PANOPTIC, "out": ttnn.DRAM_MEMORY_CONFIG}


def _torch_input(shape):
    torch.manual_seed(7)
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols)
    tile = (torch.arange(rows) // TILE)[:, None] * (-(-cols // TILE)) + (torch.arange(cols) // TILE)[None, :]
    return (x * torch.pow(2.0, (tile % 16 - 8).float())).reshape(shape).to(torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize(
    "case",
    [P2_TO_L1, P1_PANOPTIC, P1_OTHER_SPEC, P1_TO_DRAM],
    ids=["p2_to_l1", "p1_panoptic", "p1_other_spec", "p1_to_dram"],
)
def test_dispatch_outside_phase0(device, case):
    size = device.compute_with_storage_grid_size()
    if any(x1 >= size.x or y1 >= size.y for _, (x1, y1) in case["grid"]):
        pytest.skip("shard grid exceeds device grid")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in case["grid"]]
    )
    in_mc = ttnn.MemoryConfig(
        case["scheme"], ttnn.BufferType.L1, ttnn.ShardSpec(grid, case["shard"], case["orientation"])
    )
    tt_input = ttnn.from_torch(
        _torch_input(case["shape"]), dtype=case["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mc
    )
    expected = ttnn.to_torch(tt_input)

    roles = role_dump(tt_input)
    assert sum(r["shard"] is not None for r in roles) == len({r["shard"] for r in roles} - {None})

    _check_structure(tt_input, case["out"])
    for _ in range(2):  # miss, then the cached-descriptor hit
        tt_output = _dispatch(tt_input, case["out"])
        assert tt_output.memory_config().buffer_type == case["out"].buffer_type
        actual = ttnn.to_torch(tt_output)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        differ = _bits(actual) != _bits(expected)
        assert int(differ.sum()) == 0, f"{int(differ.sum())} elements differ"
        ttnn.deallocate(tt_output)
    assert torch.equal(_bits(ttnn.to_torch(tt_input)), _bits(expected))
    ttnn.deallocate(tt_input)
