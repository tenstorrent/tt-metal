# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tail_engine (K3's fused tail: latent rmsnorm + up projection + the shared expert's add, one program a chip) on the
LoudBox 2x4 against fp32 torch, each call its own inputs; with the profiler (TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1) the program's device time over traced replays."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import KIMI_K3
from models.demos.moe_block.tests.test_moe_block import pcc, replay_device_times
from models.demos.moe_block.tt.tail_engine import TailEngine

# (test-only: MOE_MESH "1x4" with MOE_FABRIC "torus_x" - one 2 x 2 square of the LoudBox as a TP ring of 4, the
# p150_x4_ring_1x4 descriptor and that square's TT_VISIBLE_DEVICES; default the 2 x 4 lines)
R, C = (int(v) for v in os.environ.get("MOE_MESH", "2x4").split("x"))
_FABRIC = {"torus_x": "FABRIC_2D_TORUS_X"}.get(os.environ.get("MOE_FABRIC", ""))


def _fab(**kw):
    p = fabric2d_device_params(**kw)
    if _FABRIC:
        p["fabric_config"] = getattr(ttnn.FabricConfig, _FABRIC)
    return p


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(KIMI_K3.latent_dim), l1_small_size=0))],
    indirect=True,
)
@pytest.mark.parametrize("ROWS", [640, 1024])  # (rows a mesh row: chunk 1280 / 2048 on the 2x4)
@pytest.mark.parametrize("LG", [3], ids=lambda g: f"lg{g}")  # (the gather's landed-share groups: 1 2 3 11 even, traced)
@pytest.mark.timeout(1800)
def test_tail_engine(mesh_device, device_params, ROWS, LG):
    m = KIMI_K3
    H, Hr, Ho = m.emb_dim, m.latent_dim, m.emb_dim // C
    g_ = torch.Generator().manual_seed(0)
    w_up = torch.randn(H, Hr, generator=g_) * 0.02
    gn = 1.0 + torch.randn(Hr, generator=g_) * 0.1
    topo = ttnn.Topology.Ring if _FABRIC else ttnn.Topology.Linear  # (the TP ring of 4: its ring schedule)
    eng = TailEngine(
        mesh_device,
        rows=ROWS,
        Hr=Hr,
        Ho=Ho,
        chunks=C,
        eps=m.norm_eps,
        topology=topo,
        land_group=LG,
    )
    logger.info(
        f"tail_engine: {ROWS} rows, grid {eng.GX} x {eng.GY} (K halves of {eng.KH} tiles), column widths "
        f"{sorted(set(eng.pn))}, steps a pair {eng.nst} (own {eng.nown}), ring slots {eng.NSLOT}, arena "
        f"{eng.arena_bytes >> 10} KB"
    )
    w = eng.upload([w_up.T[:, (d % C) * Ho : (d % C + 1) * Ho].contiguous() for d in range(R * C)], gn)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    calls = int(os.environ.get("TAIL_CALLS", "3"))
    S = ROWS // C  # (a chip's owned rows: the gather brings its mesh row's others)

    def inputs(scale):
        y = (torch.randn(R, C, S, Hr, generator=g_) * scale).to(torch.bfloat16)
        se = (torch.randn(R, C, ROWS, Ho, generator=g_) * 0.5).to(torch.bfloat16)
        # (row-major: the combine's output layout - the engine tilizes its own rows)
        yt = ttnn.from_torch(
            y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        st = ttnn.from_torch(se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        return y, se, yt, st

    for call in range(calls):
        y, se, yt, st = inputs(1.0 + call)
        out = eng.run(yt, w, st)
        dev = ttnn.to_torch(out, mesh_composer=composer).float()
        worst = 1.0
        for r in range(R):
            zf = torch.cat([y[r, c] for c in range(C)]).float()  # (the mesh row's rows, TP order)
            sc = torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + m.norm_eps)
            for c in range(C):
                ref = ((zf * gn) @ w_up.T[:, c * Ho : (c + 1) * Ho]) * sc + se[r, c].float()
                p_ = pcc(dev[r, c], ref)
                if p_ < 0.999 and worst == 1.0:
                    logger.info(
                        f"chip ({r}, {c}): per 32-row block PCC "
                        f"{[round(pcc(dev[r, c][i:i + 32], ref[i:i + 32]), 3) for i in range(0, ROWS, 32)]}"
                    )
                worst = min(worst, p_)
        logger.info(f"TAIL call {call}: PCC min over chips {worst:.6f}")
        assert worst > 0.999, f"call {call}: PCC {worst:.6f}"
        for t in (yt, st, out):
            ttnn.deallocate(t)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        # (traced: one capture of a call, replayed back to back - no host launch skew between the chips' programs)
        y, se, yt, st = inputs(1.0)
        out = eng.run(yt, w, st)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        eng.run(yt, w, st, out)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        replay_device_times(mesh_device, tid, f"tail {ROWS} lg{LG}", batches=(20, 20))
        ttnn.release_trace(mesh_device, tid)
        for t in (yt, st, out):
            ttnn.deallocate(t)
