# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blocking-knob coverage for g6_typecast (paths the acceptance test does not reach at Phase 0 defaults).

- BLOCK_TILES_CAP turned down -> several blocks per core, effective depth 2, ragged last block.
- bfloat8_b with W % 16 != 0 -> reader zeroes padding columns (mask_pad_columns).
Shapes are production shapes of feature_spec (BB-22) plus the masking cases the design requires.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

op_module = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")
pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")

MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def _bfp8_check(out, x):
    host = ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    pad = (-x.shape[-1]) % 16

    def g(t):
        t = t.to(torch.float64)
        if pad:
            t = torch.nn.functional.pad(t, (0, pad))
        return t.reshape(-1, 16)

    a, h, e = g(out), g(host), g(x)
    _, exp = torch.frexp(e.abs().amax(dim=1, keepdim=True))
    step = torch.pow(2.0, (exp - 7).to(torch.float64))
    bad = (a - h).abs() > step
    assert int(bad.sum()) == 0, f"{int(bad.sum())} values beyond one bfp8 step"


def _run(device, shape, dst, buffer):
    torch.manual_seed(0)
    x = (torch.randn(shape) * 100).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=MEM[buffer])
    out = ttnn.to_torch(g6_typecast(tx, dst))
    if dst == ttnn.bfloat8_b:
        _bfp8_check(out.to(torch.float32), x)
    elif dst == ttnn.float32:
        assert torch.equal(out.to(torch.float32).view(torch.int32), x.to(torch.float32).view(torch.int32))
    else:
        assert torch.equal(out.to(torch.int64), x.to(torch.int32).to(torch.int64))


@pytest.mark.parametrize("cap", [1, 3, 4, 7])
@pytest.mark.parametrize(
    "shape, dst, buffer",
    [
        ((16, 4096, 12), ttnn.float32, "dram"),  # Q2185: 16/15 tiles per core -> ragged tails
        ((1, 1, 3844, 256), ttnn.bfloat8_b, "dram"),  # Q2187
        ((1, 32, 64, 256), ttnn.bfloat8_b, "l1"),  # P22
        ((1, 1, 32, 128), ttnn.int32, "dram"),  # Q2229
    ],
    ids=["Q2185", "Q2187", "P22", "Q2229"],
)
def test_multi_block(device, monkeypatch, shape, dst, buffer, cap):
    monkeypatch.setattr(pd, "BLOCK_TILES_CAP", cap)
    monkeypatch.setattr(op_module, "_PROGRAM_CACHE", {})
    _run(device, shape, dst, buffer)
    plan = op_module.last_plan
    assert plan["block_tiles"] <= cap
    if plan["core_tiles_max"] > plan["block_tiles"]:
        assert plan["effective_depth"] == min(pd.BUFFER_DEPTH, -(-plan["core_tiles_max"] // plan["block_tiles"]))


@pytest.mark.parametrize("shape", [(1, 1, 64, 12), (1, 1, 33, 40), (1, 2, 64, 72)], ids=["w12", "w40", "w72"])
def test_bf8b_pad_column_mask(device, monkeypatch, shape):
    monkeypatch.setattr(op_module, "_PROGRAM_CACHE", {})
    _run(device, shape, ttnn.bfloat8_b, "dram")
    assert op_module.last_plan["mask_pad_columns"]
