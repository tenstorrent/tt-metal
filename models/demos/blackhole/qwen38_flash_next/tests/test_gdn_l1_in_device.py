# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GDN step reading the in-projection matmul's L1 width-sharded output directly (QWEN38_GDN_L1_IN=1, which
drops the ShardedToInterleaved between the matmul and gdn_step) vs today's chain (S2I to an L1 interleaved row tile, then the
step), on the TP=8 mesh: random hidden row and weights through the real DRAM-sharded matmul program, then the step (default stage
of the conv split) on both forms with identical ring slots / state / taps / constants; the output tile(s), the new state and the
newest ring slot must match BIT FOR BIT on every die, outputs sentinel-filled so a missing write cannot hide.  Also the traced time
per layer step of the two forms (S2I included for today's).
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 pytest this_file -s."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("GL_STEPS", "10"))
REPLAYS = int(os.environ.get("GL_REPLAYS", "100"))
RESULTS = os.environ.get("GL_RESULTS", "/tmp/gl_results.jsonl")
BF16, FP32, DRAM = ttnn.bfloat16, ttnn.float32, ttnn.DRAM_MEMORY_CONFIG
HIDDEN, WIDTH = 2560, gs.PROJECTION_WIDTH
SENTINEL = 3.25


def up(mesh, tensor, dtype, memory=DRAM):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=memory,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def per_device(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


class Rig:
    def __init__(self, mesh, rows, seed):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.rows = mesh, rows
        workers = 2
        self.act_cfg, self.prog = dm.dram_sharded_matmul_configs(
            mesh, HIDDEN, WIDTH, num_cores=8, num_workers_per_dram_bank=workers
        )
        w = torch.randn(1, 1, HIDDEN, WIDTH, generator=g) * 0.03
        self.weight = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dm.dram_sharded_weight_memory_config(mesh, HIDDEN, WIDTH, num_workers_per_dram_bank=workers),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.hidden = [
            up(mesh, torch.randn(1, 1, rows, HIDDEN, generator=g).to(torch.bfloat16), BF16) for _ in range(STEPS)
        ]
        self.taps = [
            up(mesh, (torch.randn(1, 1, 1, gs.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16), BF16) for _ in range(4)
        ]
        dt = (torch.randn(gs.HEADS, generator=g) * 0.5).reshape(1, 1, 1, -1)
        na = (-torch.exp(torch.randn(gs.HEADS, generator=g) * 0.5)).reshape(1, 1, 1, -1)
        self.constants = gs.constant_tiles(mesh, up(mesh, dt, FP32), up(mesh, na, FP32))
        self.norm = up(mesh, (1.0 + 0.1 * torch.randn(1, 1, 1, gs.HEAD_DIM, generator=g)).to(torch.bfloat16), BF16)
        self.state0 = torch.randn(rows, gs.HEADS, gs.HEAD_DIM, gs.HEAD_DIM, generator=g) * 0.05
        self.older = [
            [(torch.randn(1, 1, rows, gs.QKV_WIDTH, generator=g)).to(torch.bfloat16) for _ in range(3)]
            for _ in range(STEPS)
        ]

    def copy(self):
        """One form's mutable tensors: state, sentinel-filled out and newest."""
        m, r = self.mesh, self.rows
        return dict(
            state=up(m, self.state0, FP32),
            out=up(m, torch.full((1, 1, r, gs.VALUE_WIDTH), SENTINEL), BF16),
            newest=up(m, torch.full((1, 1, r, gs.QKV_WIDTH), SENTINEL), BF16),
        )

    def project(self, i):
        act = ttnn.to_memory_config(self.hidden[i], self.act_cfg)
        ws = ttnn.linear(
            act,
            self.weight,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.prog,
            compute_kernel_config=self.cfg,
        )
        ttnn.deallocate(act)
        return ws

    def step(self, projected, older, mut):
        gs.run(projected, older, mut["newest"], self.taps, self.constants, self.norm, mut["state"], mut["out"])


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 16 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("rows", [1])
def test_sharded_read_equals_interleaved(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    rig = Rig(mesh, rows, 500 + rows)
    a, b = rig.copy(), rig.copy()
    for i in range(STEPS):
        older = [up(mesh, t, BF16) for t in rig.older[i]]
        ws = rig.project(i)
        logger.info(f"projected_ws layout {ws.memory_config()}") if i == 0 else None
        inter = ttnn.to_memory_config(ws, ttnn.L1_MEMORY_CONFIG)
        rig.step(inter, older, a)
        rig.step(ws, older, b)
        ttnn.synchronize_device(mesh)
        for what in ("out", "state", "newest"):
            x, y = per_device(a[what]), per_device(b[what])
            assert all(torch.equal(p, q) for p, q in zip(x, y)), f"step {i}: {what} differs"
            assert all(torch.equal(x[0], p) for p in x)
        assert not (per_device(b["out"])[0][..., :rows, :] == SENTINEL).all()
        for t in (ws, inter, *older):
            ttnn.deallocate(t)
    logger.info(f"rows={rows}: sharded read bit-identical over {STEPS} steps")
    # trace timing per layer step: (S2I + step) vs (step on the shard)
    older = [up(mesh, t, BF16) for t in rig.older[0]]
    ws = rig.project(0)

    def traced(call, short=4, long=12):
        def capture(n):
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(n):
                call()
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

        call()
        s, l = capture(short), capture(long)
        try:
            ts, tl = wall(s), wall(l)
        finally:
            ttnn.release_trace(mesh, s)
            ttnn.release_trace(mesh, l)
        return round((tl - ts) / (long - short), 2)

    def today():
        inter = ttnn.to_memory_config(ws, ttnn.L1_MEMORY_CONFIG)
        rig.step(inter, older, a)
        ttnn.deallocate(inter)

    def direct():
        rig.step(ws, older, b)

    times = {"today_s2i_plus_step": [], "sharded_step": []}
    for _ in range(2):
        times["today_s2i_plus_step"].append(traced(today))
        times["sharded_step"].append(traced(direct))
    logger.info(f"rows={rows} us per layer: {times}")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(dict(rows=rows, us=times)) + "\n")
