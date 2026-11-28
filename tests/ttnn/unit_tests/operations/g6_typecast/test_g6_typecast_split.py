# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 — R1 sub-tile split (`tile_split`): few-tile SFPU-arm cells run one unit of NF faces of one tile per
core on a tiny-tile CB pair (quarter = 16x16, half = 32x16). Pins: regime + faces per core, every SFPU arm,
tails (dead faces dropped), witnesses spread over every face, L1, and the whole-tile fallback when units do
not fit the grid. Bit-exact vs torch (uint8: low byte of the truncation, as the golden oracle)."""

import pytest
import torch
import ttnn

import importlib

from ttnn.operations.g6_typecast import g6_typecast

op_mod = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
WITNESSES = [33685505, -33685505, 2**24 + 2**16 + 1, 257, -(2**31), 2**31 - 1, 2**24 + 1, 2**24 + 3, 0, -1]


def _input(shape, src, seed):
    g = torch.Generator().manual_seed(seed)
    if src == ttnn.int32:
        x = torch.randint(-(2**31), 2**31, shape, generator=g, dtype=torch.int64).to(torch.int32)
        flat = x.view(-1)
        # witnesses in every face of the first tile: rows 0, 1, 15 and 16, 17, 31, columns 0 / 16 / 31
        pos = [(r, c) for r in (0, 1, 15, 16, 17, 31) for c in (0, 16, 31)]
        w = shape[-1]
        for i, (r, c) in enumerate(pos):
            if r < (shape[-2] if len(shape) > 1 else 1) and c < w:
                flat[r * w + c] = WITNESSES[i % len(WITNESSES)]
        return x
    if src == ttnn.uint16:
        return torch.randint(0, 2**16, shape, generator=g, dtype=torch.int64).to(torch.int32)
    if src == ttnn.float32:
        return torch.randn(shape, generator=g, dtype=torch.float32) * 1000
    return (torch.randn(shape, generator=g, dtype=torch.float32) * 100).to(torch.bfloat16)


def _expected(x, src, dst):
    if dst == ttnn.bfloat16:
        return x.to(torch.float32).to(torch.bfloat16)
    if dst == ttnn.int32:
        return x.to(torch.int32) if src != ttnn.uint16 else (x.to(torch.int64) & 0xFFFF).to(torch.int32)
    if dst == ttnn.uint8:
        return x.to(torch.float32).to(torch.int64) & 0xFF
    raise AssertionError(dst)


CASES = [
    # id, shape, src, dst, mem, expected regime, faces per core (None = whole tiles)
    ("Q2228_i32_bf16_quarter", (1, 1, 32, 128), ttnn.int32, ttnn.bfloat16, DRAM, "tile_split", 1),
    ("i32_bf16_tails", (3, 5, 33, 45), ttnn.int32, ttnn.bfloat16, DRAM, "tile_split", 2),
    ("Q2229_bf16_i32", (1, 1, 32, 128), ttnn.bfloat16, ttnn.int32, DRAM, "tile_split", 1),
    ("Q2231_u16_i32", (1, 1, 32, 128), ttnn.uint16, ttnn.int32, DRAM, "tile_split", 1),
    ("F318_f32_bf16_half", (1, 256, 128), ttnn.float32, ttnn.bfloat16, DRAM, "tile_split", 2),
    ("f32_bf16_w12_l1", (2, 40, 12), ttnn.float32, ttnn.bfloat16, L1, "tile_split", 1),
    ("bf16_u8_small", (1, 1, 64, 96), ttnn.bfloat16, ttnn.uint8, DRAM, "tile_split", 1),
    ("i32_bf16_whole", (1, 1, 512, 512), ttnn.int32, ttnn.bfloat16, DRAM, "tile_stream", None),
]


@pytest.mark.parametrize("case_id,shape,src,dst,mem,regime,nf", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast_split(device, case_id, shape, src, dst, mem, regime, nf):
    x = _input(shape, src, seed=11)
    tt_x = ttnn.from_torch(x, dtype=src, layout=TILE, device=device, memory_config=mem)
    out = ttnn.to_torch(g6_typecast(tt_x, dst))
    plan = op_mod.last_plan
    assert plan["regime"] == regime, plan
    if nf is not None:
        assert plan["faces_per_core"] == nf, plan
    exp = _expected(x, src, dst)
    if dst == ttnn.bfloat16:
        bad = out.view(torch.int16) != exp.view(torch.int16)
    else:
        bad = (out.to(torch.int64) & (0xFF if dst == ttnn.uint8 else 0xFFFFFFFF)) != (
            exp.to(torch.int64) & (0xFF if dst == ttnn.uint8 else 0xFFFFFFFF)
        )
    assert int(bad.sum()) == 0, f"{int(bad.sum())}/{bad.numel()} differ"
