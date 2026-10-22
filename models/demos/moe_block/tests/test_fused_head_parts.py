# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE head's pieces on the LoudBox 2x4, one at a time (MOE_BLOCK_WORKLOG.md): the chunked matmul engine
(tt/lin2c.py) alone. Times need TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import ExpertActivation, act_host
from models.demos.moe_block.tests.test_tp_collectives import _time

R, C = 2, 4

# name: (K, N (gate / up pairs: 2 I), glu, GX)
LIN = {
    "glm_gu": (4096, 1024, "silu", 11),
    "glm_down": (512, 4096, None, 11),
    "k3_gu": (7168, 3072, "situ", 11),
    "k3_down": (1536, 7168, None, 11),
    "k3_gu_raw": (7168, 3072, None, 11),
    "glm_gu_raw": (4096, 1024, None, 11),
    "glm_gu_raw12": (4096, 1024, None, 12),
}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2c(mesh_device, device_params):
    from models.demos.moe_block.tt.lin2c import Lin2c, glu_weight

    GY, MC = 10, int(os.environ.get("L2C_MC", "8"))
    rows = 32 * GY * MC
    for name in os.environ.get("L2C_CASES", ",".join(LIN)).split(","):
        K, N, glu, GX = LIN[name]
        eng = Lin2c(mesh_device, cx0=0, GX=GX, GY=GY, MC=MC, K=K, N=N, glu=glu)
        g = torch.Generator().manual_seed(3)
        x = torch.randn(rows, K, generator=g)
        ws = [torch.randn(K, N, generator=g) * 0.02 for _ in range(R * C)]
        if glu:  # (W columns [gate | up] -> tile pairs)
            ws_dev = [glu_weight(w[:, : N // 2], w[:, N // 2 :]) for w in ws]
        else:
            ws_dev = ws
        wl = eng.upload(ws_dev)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: eng.run_alone(xt, wl))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        ps = []
        for d in range(R * C):
            ref = x @ ws[d]
            if glu:
                ref = act_host(
                    ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu"),
                    ref[:, : N // 2],
                    ref[:, N // 2 :],
                )
            ps.append(pcc(ys[d], ref))
            if d == 0 and os.environ.get("L2C_DIAG"):  # (per chunk x column-group block)
                M_ = 32 * GY
                for m in range(MC):
                    row = [
                        round(
                            pcc(ys[0][m * M_ : (m + 1) * M_, c0 : c0 + 32], ref[m * M_ : (m + 1) * M_, c0 : c0 + 32]), 3
                        )
                        for c0 in range(0, ys[0].shape[1], 32)
                    ]
                    logger.info(f"L2C diag chunk {m}: per output tile column {row}")
        flops = 2 * rows * K * N
        logger.info(
            f"L2C {name} M {rows} K {K} N {N} glu {glu} GX {GX}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)


# name: (H, 2 I a chip (gate / up), glu, gate / up columns, down columns)
CHAIN = {
    "glm": (4096, 1024, "silu", 8, 4),
    "k3": (7168, 3072, "situ", 8, 4),
}


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2c_chain(mesh_device, device_params):
    """The shared expert as two chained lin2c engines in one program: gate / up + GLU on columns 0 .., h in DRAM,
    down on the next columns, chunk by chunk (the down engine's x senders wait for the gate / up row's notifies)."""
    from models.demos.moe_block.tt.lin2c import Lin2c, glu_weight

    GY, MC = 10, int(os.environ.get("L2C_MC", "8"))
    rows = 32 * GY * MC
    logger.info(f"L2C grid {mesh_device.compute_with_storage_grid_size()}")
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, g1, g2 = CHAIN[name]
        if os.environ.get("L2C_SPLIT"):
            g1, g2 = map(int, os.environ["L2C_SPLIT"].split(":"))
        I = N // 2
        gu = Lin2c(mesh_device, cx0=0, GX=g1, GY=GY, MC=MC, K=H, N=N, glu=glu)
        dn = Lin2c(mesh_device, cx0=g1, GX=g2, GY=GY, MC=MC, K=I, N=H, xr_need=g1)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(rows, H, generator=g)
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        wl = gu.upload([glu_weight(w[:, :I], w[:, I:]) for w in wg])
        wld = dn.upload(wd)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: gu.run_chain(dn, xt, wl, wld))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            hh = x @ wg[d]
            ps.append(pcc(ys[d], act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]))
        flops = 2 * rows * H * N + 2 * rows * I * H
        logger.info(
            f"L2C chain {name} M {rows} H {H} I {I} cols {g1} + {g2}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2d_ref(mesh_device, device_params):
    """(debug) combine_rf.lin2d itself at a lin2c case's shape, one call of 320 rows."""
    from ttnn.bringup.combine_rf.lin2d import lin2d_prepare, upload_stacked

    K, N, _, _ = LIN[os.environ.get("L2C_CASES", "glm_gu_raw")]
    run, lay = lin2d_prepare(mesh_device, M=320, K=K, N=N, fidelity=ttnn.MathFidelity.HiFi2)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(320, K, generator=g)
    w = torch.randn(K, N, generator=g) * 0.02
    wl = upload_stacked(mesh_device, w, lay)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y = run(xt, wl)
    yt = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
    logger.info(f"L2C lin2d_ref K {K} N {N} pns {lay['pns']} NP {lay['NP']}: PCC {pcc(yt, x @ w):.5f}")
