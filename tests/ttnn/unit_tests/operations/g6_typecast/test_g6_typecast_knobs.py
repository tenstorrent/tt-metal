# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking-knob sweep for g6_typecast on production shapes (op_design.md, Perf lamps: overlap).

Each (BLOCK_TILES_CAP, depth) pair is a distinct program; run under
`scripts/run_safe_pytest.sh --profile --no-py-zones` and read DEVICE KERNEL DURATION per call.
Every call is also checked for correctness, so a knob value that breaks the ring shows up here."""

import importlib
import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

# The package re-exports the function under the module's name, so reach the modules directly.
op_module = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")
pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")

_MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

SHAPES = [
    ("Q2185", (16, 4096, 12), ttnn.float32, "dram"),
    ("Q2187", (1, 1, 3844, 256), ttnn.bfloat8_b, "dram"),
    ("P39", (1, 128, 256, 256), ttnn.bfloat8_b, "l1"),
]
CAPS = [int(v) for v in os.environ.get("G6_KNOB_CAPS", "4,8,16,32").split(",")]
DEPTHS = [int(v) for v in os.environ.get("G6_KNOB_DEPTHS", "2,3").split(",")]


@pytest.mark.parametrize("depth", DEPTHS, ids=[f"d{d}" for d in DEPTHS])
@pytest.mark.parametrize("cap", CAPS, ids=[f"cap{c}" for c in CAPS])
@pytest.mark.parametrize("case_id, shape, dtype, buffer", SHAPES, ids=[s[0] for s in SHAPES])
def test_block_sweep(device, monkeypatch, case_id, shape, dtype, buffer, cap, depth):
    monkeypatch.setattr(pd, "BLOCK_TILES_CAP", cap)
    monkeypatch.setattr(pd, "IN_DEPTH", depth)
    monkeypatch.setattr(pd, "OUT_DEPTH", depth)
    monkeypatch.setattr(op_module, "_PROGRAM_CACHE", {})

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_MEMORY_CONFIG[buffer]
    )
    out = ttnn.to_torch(g6_typecast(tx, dtype)).to(torch.float32)

    if dtype == ttnn.bfloat8_b:
        host = ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
        # One quantization step 2^(e_max - 6) of each 16-value group, as the acceptance test grades.
        groups = x.to(torch.float32).reshape(-1, 16)
        _, exp = torch.frexp(groups.abs().amax(dim=1, keepdim=True))
        step = torch.pow(2.0, (exp - 7).to(torch.float32))
        diff = (out.reshape(-1, 16) - host.to(torch.float32).reshape(-1, 16)).abs()
        assert int((diff > step).sum()) == 0
    else:
        assert torch.equal(out, x.to(torch.float32))


# Decode shapes (1-64 pages): launch/latency bound, so the lever is how many cores take part
# (op_design.md, Perf lamps: grid synchronization).
DECODE_SHAPES = [
    ("Q2227", (1, 1, 12), ttnn.float32, "l1"),
    ("Q2229", (1, 1, 32, 128), ttnn.int32, "dram"),
    ("Q2191", (16, 1, 12), ttnn.float32, "dram"),
    ("Q2203", (1, 1, 256, 256), ttnn.bfloat8_b, "dram"),
]
MIN_PAGES = [int(v) for v in os.environ.get("G6_KNOB_MIN_PAGES", "1,2,4,8,16").split(",")]


@pytest.mark.parametrize("min_pages", MIN_PAGES, ids=[f"min{m}" for m in MIN_PAGES])
@pytest.mark.parametrize("case_id, shape, dtype, buffer", DECODE_SHAPES, ids=[s[0] for s in DECODE_SHAPES])
def test_core_count_sweep(device, monkeypatch, case_id, shape, dtype, buffer, min_pages):
    monkeypatch.setattr(pd, "MIN_PAGES_PER_CORE", min_pages)
    monkeypatch.setattr(op_module, "_PROGRAM_CACHE", {})

    torch.manual_seed(0)
    x = (torch.randn(shape, dtype=torch.float32) * 1000).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_MEMORY_CONFIG[buffer]
    )
    out = ttnn.to_torch(g6_typecast(tx, dtype))

    if dtype == ttnn.bfloat8_b:
        host = ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
        groups = x.to(torch.float32).reshape(-1, 16)
        _, exp = torch.frexp(groups.abs().amax(dim=1, keepdim=True))
        step = torch.pow(2.0, (exp - 7).to(torch.float32))
        diff = (out.to(torch.float32).reshape(-1, 16) - host.to(torch.float32).reshape(-1, 16)).abs()
        assert int((diff > step).sum()) == 0
    elif dtype == ttnn.int32:
        assert torch.equal(out.to(torch.int64), x.to(torch.int32).to(torch.int64))
    else:
        assert torch.equal(out.to(torch.float32), x.to(torch.float32))
