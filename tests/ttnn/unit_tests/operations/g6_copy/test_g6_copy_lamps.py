# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-lamp sweep for g6_copy (op_design.md, Perf lamps): block/depth/stagger knobs on the
perf-focus shape, and stagger on/off over the float32 production shapes.

One op call per test id, run under scripts/run_safe_pytest.sh --profile --no-py-zones; the CSV
rows follow the collection order. Each call is checked bit for bit, so a variant that wins by
being wrong cannot pass.

Measured on Blackhole, 11x10 grid, 8 DRAM banks, (16, 12, 128, 128) fp32, median of 3:
one_block 73.9 us, one_block + stagger 67.9 us, block 4 / depth 3 + stagger 61.1 us,
block 4 / depth 2 + stagger 62.0 us, block 1 / depth 4 + stagger 70.2 us.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import describe_plan, g6_copy
from ttnn.operations.g6_copy import g6_copy_program_descriptor as plan_knobs

PERF_SHAPE = (16, 12, 128, 128)

# (MAX_BLOCK_PAGES, STREAM_DEPTH_BLOCKS, BANK_STAGGER)
VARIANTS = {
    "one_block": (None, 2, False),
    "one_block_stagger": (None, 2, True),
    "block4_depth2_stagger": (4, 2, True),
    "block4_depth3_stagger": (4, 3, True),
    "block3_depth2_stagger": (3, 2, True),
    "block2_depth2_stagger": (2, 2, True),
    "block2_depth4_stagger": (2, 4, True),
    "block1_depth4_stagger": (1, 4, True),
}
REPEATS = 3


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_g6_copy_lamp(device, monkeypatch, variant, repeat):
    max_block_pages, depth, stagger = VARIANTS[variant]
    monkeypatch.setattr(plan_knobs, "BLOCK_BYTES", None)  # MAX_BLOCK_PAGES alone sets the extent
    monkeypatch.setattr(plan_knobs, "MAX_BLOCK_PAGES", max_block_pages)
    monkeypatch.setattr(plan_knobs, "STREAM_DEPTH_BLOCKS", depth)
    monkeypatch.setattr(plan_knobs, "BANK_STAGGER", stagger)

    torch.manual_seed(0)
    src_torch = torch.randn(PERF_SHAPE, dtype=torch.float32)
    src = ttnn.from_torch(src_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    dst = ttnn.from_torch(torch.zeros(PERF_SHAPE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    plan = describe_plan(src)
    print(f"LAMP {variant}: plan={plan['plan']} block_pages={plan['block_pages']} depth={plan['cb_depth_blocks']}")

    g6_copy(src, dst)
    assert torch.equal(ttnn.to_torch(dst).view(torch.int32), src_torch.view(torch.int32))


PRODUCTION_SHAPES = [(1, 1), (1, 12, 128, 128), (2, 12, 128, 128), (8, 12, 128, 128), (16, 12, 128, 128)]


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("stagger", [False, True], ids=["no_stagger", "stagger"])
@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_stagger_production(device, monkeypatch, shape, stagger, repeat):
    monkeypatch.setattr(plan_knobs, "BANK_STAGGER", stagger)
    monkeypatch.setattr(plan_knobs, "BLOCK_BYTES", None)  # stagger measured on the one-block plan
    torch.manual_seed(0)
    src_torch = torch.randn(shape, dtype=torch.float32)
    src = ttnn.from_torch(src_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    dst = ttnn.from_torch(torch.zeros(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    assert describe_plan(src)["plan"] == "one_block"
    g6_copy(src, dst)
    assert torch.equal(ttnn.to_torch(dst).view(torch.int32), src_torch.view(torch.int32))
