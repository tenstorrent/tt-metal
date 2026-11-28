# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug coverage for the owner_writes implementation, bypassing the Phase 0 SUPPORTED gate.

Calls the internal dispatch directly so the cells a refinement will open (height sharded, bfloat8_b,
L1 output) and every blocking knob (write lanes, block extents) are proven on device before they are
claimed. Also covers the short-last-shard slot stride with more than one tile row, which no production
spec exercises. DO NOT DELETE.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import describe_plan
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _dispatch
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs

TILE = 32

P2_FLUX = dict(
    shape=(1, 5120),
    dtype=ttnn.bfloat16,
    layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(32, 160),
    grid=(((0, 0), (7, 3)),),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.DRAM_MEMORY_CONFIG,
)
P1_PANOPTIC = dict(
    shape=(1, 128, 256, 32),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(320, 32),
    grid=(((0, 0), (10, 8)), ((0, 9), (3, 9))),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.L1_MEMORY_CONFIG,
)
P2_OTHER = {**P2_FLUX, "shard": (32, 96), "grid": (((0, 0), (8, 5)),), "orientation": ttnn.ShardOrientation.COL_MAJOR}
P1_OTHER = {
    **P1_PANOPTIC,
    "shard": (352, 32),
    "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
    "orientation": ttnn.ShardOrientation.COL_MAJOR,
}
# 2 x 10 tiles width sharded 3 tile cols per shard: the last shard is 2 x 1 and keeps slot stride 3.
WIDTH_TWO_ROWS_SHORT_LAST = dict(
    shape=(1, 1, 64, 320),
    dtype=ttnn.bfloat16,
    layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(64, 96),
    grid=(((0, 0), (3, 0)),),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.DRAM_MEMORY_CONFIG,
)
# d2 = 40 pads to 64 per image: 4 tile rows, height sharded 3 tile rows per shard (last shard of 1).
HEIGHT_PADDED_IMAGE = dict(
    shape=(1, 2, 40, 64),
    dtype=ttnn.bfloat16,
    layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(96, 64),
    grid=(((1, 1), (2, 1)),),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.L1_MEMORY_CONFIG,
)

SCENARIOS = {
    "p2_flux": P2_FLUX,
    "p1_panoptic": P1_PANOPTIC,
    "p2_other": P2_OTHER,
    "p1_other": P1_OTHER,
    "p2_flux_to_l1": {**P2_FLUX, "out": ttnn.L1_MEMORY_CONFIG},
    "width_two_rows_short_last": WIDTH_TWO_ROWS_SHORT_LAST,
    "height_padded_image": HEIGHT_PADDED_IMAGE,
}
KNOBS = {
    "default": Knobs(),
    "two_lanes": Knobs(num_write_lanes=2),
    "small_blocks": Knobs(block_tile_rows=3, block_tile_cols=2),
    "two_lanes_unit_blocks": Knobs(num_write_lanes=2, block_tile_rows=1, block_tile_cols=1),
}


def _input(device, sc):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in sc["grid"]:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip("grid exceeds device")
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in sc["grid"]])
    mc = ttnn.MemoryConfig(sc["layout"], ttnn.BufferType.L1, ttnn.ShardSpec(crs, sc["shard"], sc["orientation"]))
    shape = sc["shape"]
    rows, cols = math.prod(shape[:-1]), shape[-1]
    # Distinct value per element position and a per-tile scale, so a moved tile or a regrouped bfp8 tile shows.
    x = torch.randn(rows, cols)
    t = (torch.arange(rows) // TILE)[:, None] * math.ceil(cols / TILE) + (torch.arange(cols) // TILE)[None, :]
    x = (x * torch.pow(2.0, (t % 16 - 8).float())).reshape(shape).to(torch.bfloat16)
    tt = ttnn.from_torch(x, dtype=sc["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    expected = x if sc["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt)
    return tt, expected


@pytest.mark.parametrize("knobs", list(KNOBS), ids=list(KNOBS))
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_dispatch_bit_exact(device, name, knobs):
    sc = SCENARIOS[name]
    tt_in, expected = _input(device, sc)
    out = _dispatch(tt_in, sc["out"], KNOBS[knobs])
    got = ttnn.to_torch(out)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().buffer_type == sc["out"].buffer_type
    assert out.dtype == sc["dtype"]
    differ = got.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    assert int(differ.sum()) == 0, f"{int(differ.sum())} elements differ"
    ttnn.deallocate(out)
    ttnn.deallocate(tt_in)


def test_describe_plan_production(device):
    tt_in, _ = _input(device, P1_PANOPTIC)
    plan = describe_plan(tt_in, P1_PANOPTIC["out"])
    writers = [c for c in plan["cores"] if c["role"] == "writer"]
    assert plan["plan"] == "owner_writes"
    assert plan["num_shards"] == 103 and len(writers) == 103
    assert writers[-1]["core"] == (3, 9) and writers[-1]["block"] == (1020, 0, 4, 1)
    ttnn.deallocate(tt_in)
