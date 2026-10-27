# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tail_a2a (the non-latent models' fused tail: tilize + the shared expert's add + the TP all-to-all, one program a
chip) on the LoudBox 2x4 (or a TP ring of 4: MOE_MESH 1x4 MOE_FABRIC torus_x) against torch - to a bf16 ulp (the
add's rounding; the data moves as is) - each call its own inputs, then traced replays back to back (TAIL_CALLS of them, every 25th
checked: a soak), and with the profiler the program's device time against the ops tail (tilize, add, all-to-all)."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.moe_block.tests.test_moe_block import replay_device_times
from models.demos.moe_block.tests.test_tail_engine import _FABRIC, C, R, _fab
from models.demos.moe_block.tt.tail_a2a import TailA2A

MODELS = {"k27": 7168, "glm53": 6144, "glmf": 4096}


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(7168), l1_small_size=0, trace_region_size=16 << 20))],
    indirect=True,
)
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("ROWS", [640, 1024])
@pytest.mark.timeout(3600)
def test_tail_a2a(mesh_device, device_params, model, ROWS):
    H = MODELS[model]
    S, Ho = ROWS // C, H // C
    topo = ttnn.Topology.Ring if _FABRIC else ttnn.Topology.Linear
    tail = TailA2A(mesh_device, rows=ROWS, H=H, topology=topo, num_links=2)
    ports = {k: (c.x, c.y) for k, c in next(iter(tail.chips.values()))["ports"].items()}
    logger.info(f"tail_a2a {model} {ROWS} rows: {tail.NW} workers of {tail.W} tiles, ports {ports}, RP {tail.RP}")
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    g_ = torch.Generator().manual_seed(0)

    def inputs():
        y = torch.randn(R, C, S, H, generator=g_).to(torch.bfloat16)
        se = torch.randn(R, C, S, H, generator=g_).to(torch.bfloat16)
        yt = ttnn.from_torch(
            y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        st = ttnn.from_torch(se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        return y, se, yt, st

    def check(out, y, se, tag):
        dev = ttnn.to_torch(out, mesh_composer=composer)
        z = (y.float() + se.float()).to(torch.bfloat16)  # (the device's bf16 add)
        bad = 0
        for r in range(R):
            for q in range(C):
                ref = torch.cat([z[r, p][:, q * Ho : (q + 1) * Ho] for p in range(C)])
                diff = (dev[r, q].float() - ref.float()).abs() - ref.float().abs() / 128  # (a bf16 ulp)
                if diff.max() > 1e-6:
                    rows = sorted({int(i) // 32 for i in torch.nonzero(diff.amax(-1) > 1e-6).flatten()})
                    logger.info(f"{tag}: chip ({r}, {q}) off at row tiles {rows[:12]}, max {diff.max():.3g}")
                    bad += 1
        assert bad == 0, f"{tag}: {bad} chips off"

    for call in range(3):
        y, se, yt, st = inputs()
        out = tail.run(yt, st)
        check(out, y, se, f"call {call}")
        for t in (yt, st, out):
            ttnn.deallocate(t)
    logger.info(f"TAIL_A2A {model} {ROWS}: 3 eager calls right")
    y, se, yt, st = inputs()
    out = tail.run(yt, st)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tail.run(yt, st, out)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    n = int(os.environ.get("TAIL_CALLS", "50"))
    for i in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        if i % 25 == 24 or i == n - 1:
            ttnn.synchronize_device(mesh_device)
            check(out, y, se, f"replay {i + 1}")
    logger.info(f"TAIL_A2A {model} {ROWS}: {n} traced replays right")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        replay_device_times(mesh_device, tid, f"tail_a2a {model} {ROWS}", batches=(20, 20))
    ttnn.release_trace(mesh_device, tid)
    for t in (yt, st, out):
        ttnn.deallocate(t)
