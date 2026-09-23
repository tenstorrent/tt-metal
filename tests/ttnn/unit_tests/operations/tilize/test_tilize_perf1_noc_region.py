"""noc_region bake-off (perf-part-optimizer, experiment only; run with run_safe_pytest.sh [--profile]).

NOCR_RULES="unpatched,default,swap_all,..."  (rules in perf_experiments/noc_region/noc_region.py)
NOCR_CASES="focus,..."                        (keys of CASES below)
NOCR_ASSIGN="-,oracle,..."                   (secondary lever: host row split, see noc_region_split.py; - = op's split)
Every cell is bit-exact checked (torch.equal) against the input.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/noc_region"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"noc_region_{name}", EXP / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


nr = _load("noc_region")


def _hs(grid_x, grid_y, shard):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


# id: (shape, in memory config, out memory config or None)
CASES = {
    "focus": ((1, 1, 16384, 64), "dram", None),
    "h32k_w64": ((1, 1, 32768, 64), "dram", None),
    "h16k_w32": ((1, 1, 16384, 32), "dram", None),
    "h8k_w256": ((1, 1, 8192, 256), "dram", None),
    "h2k_w1024": ((1, 1, 2048, 1024), "dram", None),
    "h128_w64": ((1, 1, 128, 64), "dram", None),
    "h32_w2048": ((1, 1, 32, 2048), "dram", None),
    "l1_h4k_w64": ((1, 1, 4096, 64), "l1", "l1"),
    "hs_h2k_w512": ((1, 1, 2048, 512), "hs", "dram"),
}
RULES = os.environ.get("NOCR_RULES", "unpatched,default,swap_all").split(",")
CASE_IDS = os.environ.get("NOCR_CASES", "focus").split(",")
ASSIGNS = os.environ.get("NOCR_ASSIGN", "-").split(",")


def _mc(kind, shape):
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if kind == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if kind == "hs":
        return _hs(8, 8, (32, 512))  # the r3 guard's sharded_accessor input
    raise ValueError(kind)


@pytest.mark.parametrize("assign", ASSIGNS)
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("case", CASE_IDS)
def test_noc_region(device, monkeypatch, case, rule, assign):
    shape, in_kind, out_kind = CASES[case]
    if assign != "-" and (in_kind == "hs" or out_kind == "hs"):
        pytest.skip("row rebalance: a resident shard fixes the core assignment")
    if assign != "-":
        _load("noc_region_split").install(monkeypatch, device, assign, rule_name=rule)
    elif rule != "unpatched":
        nr.install(monkeypatch, device, rule)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_mc(in_kind, shape)
    )
    kw = {} if out_kind is None else {"memory_config": _mc(out_kind, shape)}
    out = tilize(t, **kw)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out), x), f"{case} {rule} mismatch"
    print(f"NOCR {case} {rule} {assign} ok")
