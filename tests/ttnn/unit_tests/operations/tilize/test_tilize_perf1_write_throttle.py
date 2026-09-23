"""write_throttle bake-off harness (run with run_safe_pytest.sh [--profile]).

TILIZE_WT="base,ws4,rs2@BANK_COALESCE_STAGE_DEPTH=1,..."  variant = kernel dir token (gen.py name,
  '+'-joined tokens) optionally followed by '@KNOB=int[;KNOB=int]' host-knob overrides.
TILIZE_WT_CASES="1x1x16384x64,1x1x4096x64:l1:l1,1x1x2048x512:hs:dram,..."  shape[:in_mc[:out_mc]].
Every cell asserts bit-exact output (torch.equal).
"""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

HERE = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/write_throttle"
VARS = os.environ.get("TILIZE_WT", "base").split(",")
CASES = os.environ.get("TILIZE_WT_CASES", "1x1x16384x64").split(",")


def _hs(grid_x, grid_y, shard):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _mc(name):
    return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG, "hs": _hs(8, 8, (32, 512))}[name]


@pytest.mark.parametrize("variant", VARS)
@pytest.mark.parametrize("case", CASES)
def test_wt(device, monkeypatch, case, variant):
    kdir, _, knobs = variant.partition("@")
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / ("kernels_" + kdir.replace("+", "_")))
    for kv in filter(None, knobs.split(";")):
        k, v = kv.split("=")
        monkeypatch.setattr(pd, k, int(v))
    parts = case.split(":")
    shape = tuple(int(d) for d in parts[0].split("x"))
    in_mc = parts[1] if len(parts) > 1 else "dram"
    out_mc = parts[2] if len(parts) > 2 else None
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_mc(in_mc))
    out = tilize(t, memory_config=_mc(out_mc)) if out_mc else tilize(t)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out), x)
    print(f"WT {variant} {case} ok")
