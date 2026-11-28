# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1 — ROW_MAJOR regime R2 `rm_pseudo_tile_stream` and its conversion arms.

Pins: one ROW_MAJOR and one TILE case per regime (last_plan["regime"]), the new arms
(bf16->uint8 truncation, uint32->int32 / int32->uint32 keeping all 32 bits), a forced multi-block /
ragged-tail RM case (BLOCK_TILES_CAP), an L1 RM case, and the host refusals (no-arm pair, bf8b + RM).
All values bit-exact against torch.Tensor.to.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

op_mod = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")
pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

_TORCH = {ttnn.int32: torch.int32, ttnn.uint32: torch.uint32, ttnn.uint8: torch.uint8}


def make_input(shape, src, dst):
    torch.manual_seed(7)
    n = 1
    for d in shape:
        n *= d
    if src == ttnn.bfloat16:
        x = (torch.rand(n) * 255).to(torch.bfloat16)
        w = [0.0, 0.5, 1.0, 1.9921875, 127.5, 254.0, 255.0]
        k = min(len(w), n)
        x[:k] = torch.tensor(w[:k], dtype=torch.bfloat16)
    elif src == ttnn.uint32:
        x = torch.randint(0, 2**32, (n,), dtype=torch.int64).to(torch.uint32)
    else:
        x = torch.randint(-(2**31), 2**31, (n,), dtype=torch.int64).to(torch.int32)
        w = [-123456789, -1, 0, 2**31 - 1, -(2**31)]
        k = min(len(w), n)
        x[:k] = torch.tensor(w[:k], dtype=torch.int32)
    return x.reshape(shape)


def run(device, shape, src, dst, layout, mem=DRAM):
    x = make_input(shape, src, dst)
    expected = x.to(_TORCH[dst])
    tt_x = ttnn.from_torch(x, dtype=src, layout=layout, device=device, memory_config=mem)
    tt_out = g6_typecast(tt_x, dst)
    assert list(tt_out.shape) == list(shape)
    assert tt_out.dtype == dst
    assert tt_out.layout == layout
    assert tt_out.memory_config() == tt_x.memory_config()
    actual = ttnn.to_torch(tt_out)
    width = 8 if dst == ttnn.uint8 else 32
    mask = (1 << width) - 1
    a = actual.to(torch.int64) & mask
    e = expected.to(torch.int64) & mask
    bad = a != e
    n_bad = int(bad.sum())
    if n_bad:
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        detail = ", ".join(f"[{i}] got {a.reshape(-1)[i].item()} exp {e.reshape(-1)[i].item()}" for i in idx)
        raise AssertionError(f"{n_bad}/{bad.numel()} differ: {detail}")
    return op_mod.last_plan


CASES = [
    # id, shape, src, dst, layout, mem
    ("Q2189_u32_i32_rm_wide", (1, 3844), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("Q2199_u32_i32_rm_31", (1, 31), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("F321_i32_u32_rank1", (1,), ttnn.int32, ttnn.uint32, RM, DRAM),
    ("rm_multi_row_u32_i32", (3, 77, 33), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("rm_l1_bf16_u8", (2, 50, 3), ttnn.bfloat16, ttnn.uint8, RM, L1),
    ("F317_bf16_u8_rm", (1, 1024, 1024, 3), ttnn.bfloat16, ttnn.uint8, RM, DRAM),
    ("Q2230_u32_i32_tile", (1, 1, 32, 128), ttnn.uint32, ttnn.int32, TILE, DRAM),
    ("tile_i32_u32", (1, 1, 64, 96), ttnn.int32, ttnn.uint32, TILE, DRAM),
]


@pytest.mark.parametrize("case_id,shape,src,dst,layout,mem", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast_rm_arms(device, case_id, shape, src, dst, layout, mem):
    plan = run(device, shape, src, dst, layout, mem)
    if (src, dst) in pd.BIT_IDENTICAL_ARMS:
        expected_regime = "pure_copy"  # Perf 1: bit-identical arms are a byte copy in both layouts
    else:
        expected_regime = "rm_pseudo_tile_stream" if layout == RM else "tile_stream"
    assert plan["regime"] == expected_regime


@pytest.mark.parametrize("cap", [1, 3])
def test_g6_typecast_rm_multi_block(device, cap, monkeypatch):
    """Force several blocks per core (and a ragged last block) through BLOCK_TILES_CAP."""
    monkeypatch.setattr(pd, "BLOCK_TILES_CAP", cap)
    plan = run(device, (1, 400, 300, 3), ttnn.bfloat16, ttnn.uint8, RM)
    assert plan["block_tiles"] <= cap
    assert plan["core_rows_max"] > plan["rows_per_block"]


def test_g6_typecast_refusals(device, expect_error):
    x = ttnn.from_torch(
        torch.zeros(1, 64, dtype=torch.int64).to(torch.uint32), dtype=ttnn.uint32, layout=RM, device=device
    )
    with expect_error(ValueError, "no conversion arm"):
        g6_typecast(x, ttnn.uint8)  # per-axis supported, no arm
    y = ttnn.from_torch(torch.zeros(1, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=RM, device=device)
    with expect_error(ValueError, "ROW_MAJOR"):
        g6_typecast(y, ttnn.bfloat8_b)


# Perf 1 — regime R0 `pure_copy`: page mode (pages >= cores), chunk mode (one long page split over cores,
# sub-chunked when a chunk exceeds the L1 scratch), L1 buffers, both arms, ranks 1..4.
PURE_COPY_CASES = [
    ("chunk_long_row", (1, 4096), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("chunk_rank1_5000", (5000,), ttnn.int32, ttnn.uint32, RM, DRAM),
    ("chunk_row_65536", (1, 65536), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("chunk_tile_l1", (1, 1, 32, 128), ttnn.uint32, ttnn.int32, TILE, L1),
    ("page_rm_l1", (3, 77, 33), ttnn.int32, ttnn.uint32, RM, L1),
    ("page_rm_many", (2, 4, 64, 256), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("page_rm_wide_rows", (200, 4096), ttnn.uint32, ttnn.int32, RM, DRAM),
    ("page_tile_big", (1, 1, 1024, 1024), ttnn.int32, ttnn.uint32, TILE, DRAM),
    ("page_rank4_small", (2, 3, 5, 7), ttnn.int32, ttnn.uint32, RM, DRAM),
]


@pytest.mark.parametrize("case_id,shape,src,dst,layout,mem", PURE_COPY_CASES, ids=[c[0] for c in PURE_COPY_CASES])
def test_g6_typecast_pure_copy(device, case_id, shape, src, dst, layout, mem):
    plan = run(device, shape, src, dst, layout, mem)
    assert plan["regime"] == "pure_copy"
