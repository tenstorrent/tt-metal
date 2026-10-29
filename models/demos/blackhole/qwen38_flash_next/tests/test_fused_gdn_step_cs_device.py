# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the conv-split GDN step (QWEN38_GDN_CONV_SPLIT=1) against today's program on the TP=8 mesh.

Same random inputs (a realistic in-projection row, three ring slots, four taps, fp32 state, dt/A, norm weight) on two
copies: today's ``gdn_step.run`` and the split one; the gated output tile(s), the new state and the newest ring slot must
match BIT FOR BIT on every die, for STEPS consecutive steps (the state evolves in place; a semaphore left non-zero would
show).  Also times both programs in a trace: wall us per op = slope between a 4-op and a 12-op trace.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 pytest this_file -s -p conftest.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
ENV = "QWEN38_GDN_CONV_SPLIT"
STEPS = int(os.environ.get("CS_STEPS", "20"))
REPLAYS = int(os.environ.get("CS_REPLAYS", "200"))
RESULTS = Path(os.environ.get("CS_RESULTS", "/tmp/cs_device_results.jsonl"))
BF16, FP32, DRAM = ttnn.bfloat16, ttnn.float32, ttnn.DRAM_MEMORY_CONFIG


def up(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def per_device(tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def realistic(generator, *shape):
    """Mostly unit-scale values with a few columns scaled large and small (the conv inputs' dynamic range)."""
    x = torch.randn(*shape, generator=generator)
    cols = torch.rand(shape[-1], generator=generator)
    scale = torch.where(cols < 0.05, 30.0, torch.where(cols < 0.15, 1e-3, 1.0))
    return (x * scale).to(torch.bfloat16)


class Case:
    """Host values of one test case: taps, constants, norm, state, and the per-step projections and ring slots."""

    def __init__(self, rows, seed):
        g = torch.Generator().manual_seed(seed)
        self.rows = rows
        self.taps = [realistic(g, 1, 1, 1, gs.QKV_WIDTH) * 0.5 for _ in range(4)]
        self.dt_bias = (torch.randn(gs.HEADS, generator=g) * 0.5).reshape(1, 1, 1, -1)
        self.neg_exp_a = (-torch.exp(torch.randn(gs.HEADS, generator=g) * 0.5)).reshape(1, 1, 1, -1)
        self.norm = (1.0 + 0.1 * torch.randn(1, 1, 1, gs.HEAD_DIM, generator=g)).to(torch.bfloat16)
        self.state = torch.randn(rows, gs.HEADS, gs.HEAD_DIM, gs.HEAD_DIM, generator=g) * 0.05
        self.projections = [realistic(g, 1, 1, rows, gs.PROJECTION_WIDTH) for _ in range(STEPS)]
        for p in self.projections:  # a/b columns small enough for a sane gate range
            p[..., gs.A_COLUMN :] = (p[..., gs.A_COLUMN :].float().clamp(-4, 4)).to(torch.bfloat16)
        self.older = [[realistic(g, 1, 1, rows, gs.QKV_WIDTH) for _ in range(3)] for _ in range(STEPS)]


class Run:
    """Device tensors of one copy of the case."""

    def __init__(self, mesh, case):
        self.mesh, self.case, rows = mesh, case, case.rows
        self.taps = [up(mesh, t, BF16) for t in case.taps]
        self.constants = gs.constant_tiles(mesh, up(mesh, case.dt_bias, FP32), up(mesh, case.neg_exp_a, FP32))
        self.norm = up(mesh, case.norm, BF16)
        self.state = up(mesh, case.state, FP32)
        self.out = up(mesh, torch.zeros(1, 1, rows, gs.VALUE_WIDTH), BF16)
        self.newest = up(mesh, torch.zeros(1, 1, rows, gs.QKV_WIDTH), BF16)

    def step(self, i):
        c = self.case
        projected = up(self.mesh, c.projections[i], BF16)
        older = [up(self.mesh, t, BF16) for t in c.older[i]]
        gs.run(projected, older, self.newest, self.taps, self.constants, self.norm, self.state, self.out)
        return projected, older

    def read(self):
        return per_device(self.out), per_device(self.state), per_device(self.newest)


def traced_us(mesh, run_call, short=4, long=12):
    """Wall us per op: slope of the (long - short)-op traces, REPLAYS replays each, min of 3 (x24's recipe)."""

    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ops):
            run_call()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    a, b = capture(short), capture(long)
    try:
        ta, tb = wall(a), wall(b)
    finally:
        ttnn.release_trace(mesh, a)
        ttnn.release_trace(mesh, b)
    return (tb - ta) / (long - short), tb / long


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_split_equals_default(mesh_device, rows):
    assert gs.HEADS == 6, "run with MESH_DEVICE=P150x8"
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"compute grid {grid.x}x{grid.y}; load {os.getloadavg()}")
    case = Case(rows, 100 + rows)
    os.environ[ENV] = "0"  # unset = stage 1 since the default flip
    base = Run(mesh_device, case)
    os.environ[ENV] = "1"
    split = Run(mesh_device, case)
    table = []
    try:
        for i in range(STEPS):
            os.environ[ENV] = "0"  # unset = stage 1 since the default flip
            keep_b = base.step(i)
            os.environ[ENV] = "1"
            keep_s = split.step(i)
            (bo, bs, bn), (so, ss, sn) = base.read(), split.read()
            row = dict(
                step=i,
                out=all(torch.equal(x, y) for x, y in zip(bo, so)),
                state=all(torch.equal(x, y) for x, y in zip(bs, ss)),
                newest=all(torch.equal(x, y) for x, y in zip(bn, sn)),
                dies_agree=all(torch.equal(bo[0], x) and torch.equal(so[0], y) for x, y in zip(bo, so)),
                moved=bool((bs[0] != torch.from_numpy(case.state.numpy())).any()),
            )
            table.append(row)
            for t in (*keep_b[1], keep_b[0], *keep_s[1], keep_s[0]):
                ttnn.deallocate(t)
        logger.info(
            f"rows={rows} bitwise: out {all(r['out'] for r in table)} state {all(r['state'] for r in table)} "
            f"newest {all(r['newest'] for r in table)} over {STEPS} steps"
        )
        assert all(r["out"] and r["state"] and r["newest"] for r in table), table
        assert all(r["moved"] for r in table)
        assert torch.isfinite(base.read()[0][0].float()).all()

        # timing: both programs on the same inputs, in a trace (the program cache is warm from the steps above)
        projected = up(mesh_device, case.projections[0], BF16)
        older = [up(mesh_device, t, BF16) for t in case.older[0]]
        times = {}
        for name, flag in (("today", None), ("split", "1"), ("today2", None), ("split2", "1")):
            if flag is None:
                os.environ[ENV] = "0"  # unset = stage 1 since the default flip
                target = base
            else:
                os.environ[ENV] = "1"
                target = split
            call = lambda: gs.run(
                projected, older, target.newest, target.taps, target.constants, target.norm, target.state, target.out
            )
            times[name] = traced_us(mesh_device, call)
            logger.info(f"rows={rows} {name}: slope {times[name][0]:.2f} us/op, T12/12 {times[name][1]:.2f} us")
        with RESULTS.open("a") as f:
            f.write(json.dumps(dict(rows=rows, steps=STEPS, bitwise=True, times=times, load=os.getloadavg())) + "\n")
    finally:
        os.environ[ENV] = "0"  # unset = stage 1 since the default flip
