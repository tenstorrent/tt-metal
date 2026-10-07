# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass (#58725, #58726): compute-sensitivity probe (EB_R3_PROBE_INIT, one more binary init per
tile) on sharded ops with a column or scalar broadcast and an activation on the per-tile operand (rsub's NEG, the logical ops'
NEZ), and on row broadcasts whose b sits in L1 or is sharded. CI only."""
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _shard(shape, strat, grid):
    return ttnn.create_sharded_memory_config(shape, core_grid=grid, strategy=strat)


A = {"hs8": ((1, 1, 1024, 1024), ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=4)),
     "bs64": ((1, 1, 4096, 1280), ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=8, x=8))}
OPS = {
    "rsub": (lambda a, b, **kw: ttnn.rsub(a, b, **kw), lambda a, b: b - a),
    "logical_and": (lambda a, b, **kw: ttnn.logical_and(a, b, **kw), lambda a, b: torch.logical_and(a, b).float()),
    "add": (lambda a, b, **kw: ttnn.add(a, b, **kw), lambda a, b: a + b),
    "mul": (lambda a, b, **kw: ttnn.multiply(a, b, fast_and_approximate_mode=True, **kw), lambda a, b: a * b),
}
CASES = [(op, a, k, "dram") for op in ("rsub", "logical_and") for a in ("hs8", "bs64") for k in ("col", "scalar")]
CASES += [(op, a, "row", m) for op in ("add", "mul") for a in ("hs8", "bs64") for m in ("dram", "l1")]
CASES += [(op, "hs8", "row", "sharded") for op in ("add", "mul")]


@pytest.mark.parametrize("op, am, kind, bm", CASES, ids=["-".join(c) for c in CASES])
def test_probe3(device, op, am, kind, bm):
    shape, strat, grid = A[am]
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    b_shape = {"col": (1, 1, shape[2], 1), "scalar": (1, 1, 1, 1), "row": (1, 1, 1, shape[3])}[kind]
    b = torch.rand(b_shape, dtype=torch.bfloat16) - 0.5
    mca = _shard(shape, strat, grid)
    if bm == "dram":
        mcb = ttnn.DRAM_MEMORY_CONFIG
    elif bm == "l1":
        mcb = ttnn.L1_MEMORY_CONFIG
    else:
        mcb = ttnn.create_sharded_memory_config((1, 1, 32, shape[3]), core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mca)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mcb)
    f, g = OPS[op]
    for _ in range(3):
        out = f(ta, tb, memory_config=mca)
    got = ttnn.to_torch(out).float()
    ref = g(ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float())
    assert torch.allclose(got, ref, rtol=0.05, atol=0.05), float((got - ref).abs().max())
