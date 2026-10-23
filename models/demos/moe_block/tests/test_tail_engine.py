# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tail_engine (K3's fused tail: latent rmsnorm + up projection + the shared expert's add, one program a chip) on the
LoudBox 2x4 against fp32 torch, each call its own inputs; the program's device time with the profiler
(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1)."""

import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import KIMI_K3
from models.demos.moe_block.tests.test_moe_block import pcc, per_chip_programs
from models.demos.moe_block.tt.tail_engine import TailEngine

R, C = 2, 4
ROWS = 640


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(KIMI_K3.latent_dim), l1_small_size=0))],
    indirect=True,
)
@pytest.mark.timeout(1800)
def test_tail_engine(mesh_device, device_params):
    m = KIMI_K3
    H, Hr, Ho = m.emb_dim, m.latent_dim, m.emb_dim // C
    g_ = torch.Generator().manual_seed(0)
    w_up = torch.randn(H, Hr, generator=g_) * 0.02
    gn = 1.0 + torch.randn(Hr, generator=g_) * 0.1
    eng = TailEngine(mesh_device, rows=ROWS, Hr=Hr, Ho=Ho, chunks=C, eps=m.norm_eps)
    logger.info(
        f"tail_engine: grid {eng.GX} x {eng.GY} (K halves of {eng.KH} tiles), column widths {sorted(set(eng.pn))}, "
        f"arena {eng.arena_bytes >> 10} KB"
    )
    w = eng.upload([w_up.T[:, (d % C) * Ho : (d % C + 1) * Ho].contiguous() for d in range(R * C)], gn)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    calls = int(os.environ.get("TAIL_CALLS", "3"))
    for call in range(calls):
        z = (torch.randn(R, C, ROWS, Hr, generator=g_) * (1.0 + call)).to(torch.bfloat16)
        se = (torch.randn(R, C, ROWS, Ho, generator=g_) * 0.5).to(torch.bfloat16)
        zt = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        st = ttnn.from_torch(se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        out = eng.run(zt, w, st)
        y = ttnn.to_torch(out, mesh_composer=composer).float()
        worst = 1.0
        probe = 0
        for r in range(R):
            for c in range(C):
                zf = z[r, c].float()
                sc = torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + m.norm_eps)
                raw = (zf * gn) @ w_up.T[:, c * Ho : (c + 1) * Ho]
                ref = {
                    0: raw * sc + se[r, c].float(),
                    1: se[r, c].float(),
                    2: raw,
                    3: sc.expand(-1, Ho),
                    4: zf.pow(2).sum(-1, keepdim=True).expand(-1, Ho),
                }[probe]
                p_ = pcc(y[r, c], ref)
                if probe == 4 and r == 0 and c == 0:
                    logger.info(
                        f"sumsq dev/ref rows 0..7: {(y[r, c][:8, 0] / ref[:8, 0]).tolist()}; cols of row 0: {y[r, c][0, :6].tolist()}"
                    )
                if p_ < 0.999 and worst == 1.0:
                    logger.info(
                        f"chip ({r}, {c}): y nan {int(y[r, c].isnan().sum())}, |y| {y[r, c].abs().mean():.4f} vs |ref| "
                        f"{ref.abs().mean():.4f}; per 32-row block PCC "
                        f"{[round(pcc(y[r, c][i:i + 32], ref[i:i + 32]), 3) for i in range(0, ROWS, 32)]}"
                    )
                worst = min(worst, p_)
        logger.info(f"TAIL call {call}: PCC min over chips {worst:.6f}")
        assert worst > 0.999, f"call {call}: PCC {worst:.6f}"
        for t in (zt, st, out):
            ttnn.deallocate(t)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        per_chip_programs(mesh_device)
        ts = []
        for _ in range(5):
            z = torch.randn(R, C, ROWS, Hr, generator=g_).to(torch.bfloat16)
            zt = ttnn.from_torch(
                z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
            )
            st = ttnn.from_torch(
                z[..., :Ho], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
            )
            ttnn.synchronize_device(mesh_device)
            per_chip_programs(mesh_device)
            out = eng.run(zt, w, st)
            ttnn.synchronize_device(mesh_device)
            progs = [p for p in per_chip_programs(mesh_device) if p]
            ts.append(max(sum(p) for p in progs))
            for t in (zt, st, out):
                ttnn.deallocate(t)
        logger.info(
            f"TAIL device time {statistics.median(ts):.1f} us (slowest chip, median of 5; {[round(t, 1) for t in ts]})"
        )
