# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the K3 shared expert TP-sharded, piece by piece, on the LoudBox 2 x 4 (FABRIC_2D, 200G): TP over the mesh
row (4 chips) or the column (2 chips); the token all-gather and the partial reduce-scatter with bf16 / bf8 payloads
(ttnn CCLs); the shared expert as dense ttnn matmuls at the TP shapes (HiFi2, bf8 weights, mm2d configs). Device time
per program (slowest chip) and the output's error vs fp32 (own 320 rows per chip)."""
import os

import pytest
import torch
from loguru import logger

import models.demos.glm53_flash_d_p.tt.mm_configs as mmc
import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims

R, C = 2, 4
S = 320
_KEY = "DEVICE KERNEL DURATION [ns]"
_PARAMS = [pytest.param((R, C), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="2x4")]


def _times(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    rows = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        rows.append([float(p.program_analyses_results[_KEY].duration) / 1e3 for p in programs if _KEY in (p.program_analyses_results or {})])
    return rows


def _err(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a - b).norm().item() / b.norm().item(), torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_se_tp(mesh_device, device_params):
    d = K3Dims()
    H, I = d.H, d.I_shared
    g = torch.Generator().manual_seed(0)
    wg, wu, wd = (torch.randn(I, H, generator=g) * 0.02, torch.randn(I, H, generator=g) * 0.02, torch.randn(H, I, generator=g) * 0.02)
    x = torch.randn(R * C * S, H, generator=g).to(torch.bfloat16)  # (chip r * C + c holds rows (r C + c) S ..)

    def ref_se(xx):
        gs, us = xx.float() @ wg.T, xx.float() @ wu.T
        a = d.beta * torch.tanh(gs / d.beta) * torch.sigmoid(gs) * (d.linear_beta * torch.tanh(us / d.linear_beta))
        return a @ wd.T

    yref = ref_se(x)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0))
    cfg = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
    mmc.ENABLED = True

    def lin(a, w):
        return ttnn.linear(a, w, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=mmc.linear_config(a, w, ttnn.bfloat16))

    def timed(label, f, n=3):
        _times(mesh_device)
        best, out = None, None
        for _ in range(n):
            out = f()
            ttnn.synchronize_device(mesh_device)
            rows = _times(mesh_device)
            progs = [round(max(r[i] for r in rows), 1) for i in range(min(len(r) for r in rows))]
            tot = sum(progs)
            best = (tot, progs) if best is None or tot < best[0] else best
        logger.info(f"SETP {label:44s} {best[0]:8.1f} us  {best[1]}")
        return out

    def own_rows(t, rows_per_chip):
        return torch.cat([ttnn.to_torch(u).reshape(rows_per_chip, H) for u in ttnn.get_device_tensors(t)])

    for axis in [int(a) for a in os.environ.get("SETP_AXES", "1,0").split(",")]:
        G = C if axis == 1 else R  # (the TP group: a mesh row or column)
        Ig = I // G
        # chip (r, c): its group index k = c (row TP) or r (column TP): intermediate slice k Ig ..
        dims = (None, -1) if axis == 1 else (-1, None)
        dims_d = (None, -2) if axis == 1 else (-2, None)
        m2 = lambda t, dd: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device,  # noqa: E731
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                           mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=dd))
        wgt, wut, wdt = m2(wg.T.contiguous(), dims), m2(wu.T.contiguous(), dims), m2(wd.T.contiguous(), dims_d)
        for pdt, pname in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8")):
            tag = f"TP{G} ({'row' if axis == 1 else 'col'}) {pname}"
            xs = xt if pdt == ttnn.bfloat16 else ttnn.typecast(xt, pdt)
            xg = timed(f"{tag} all_gather x", lambda: ttnn.all_gather(xs, dim=0, cluster_axis=axis, num_links=2,
                                                                       topology=ttnn.Topology.Linear))
            xg16 = xg if pdt == ttnn.bfloat16 else ttnn.typecast(xg, ttnn.bfloat16)

            def se():
                gt, ut = lin(xg16, wgt), lin(xg16, wut)
                a = ttnn.situ_glu(gt, ut, d.beta, d.linear_beta)
                ttnn.deallocate(gt)
                ttnn.deallocate(ut)
                y = lin(a, wdt)
                ttnn.deallocate(a)
                return y

            y = timed(f"{tag} dense SE [{G * S}, I {Ig}]", se)
            yp = y if pdt == ttnn.bfloat16 else ttnn.typecast(y, pdt)
            ys = timed(f"{tag} reduce_scatter", lambda: ttnn.reduce_scatter(yp, 0, cluster_axis=axis, num_links=2,
                                                                            topology=ttnn.Topology.Linear))
            ys16 = ys if pdt == ttnn.bfloat16 else ttnn.typecast(ys, ttnn.bfloat16)
            got = own_rows(ys16, S)
            rel, p = _err(got, yref)
            logger.info(f"SETP {tag} output vs fp32: rel {rel * 100:.2f}%  pcc {p:.5f}")
            for t in {id(v): v for v in (xg, xg16, y, yp, ys, ys16)}.values():
                if t.is_allocated():
                    ttnn.deallocate(t)
            if pdt != ttnn.bfloat16:
                ttnn.deallocate(xs)
        for t in (wgt, wut, wdt):
            ttnn.deallocate(t)
    # local (replicated) dense reference
    rep = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device,  # noqa: E731
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    wgt, wut, wdt = rep(wg.T.contiguous()), rep(wu.T.contiguous()), rep(wd.T.contiguous())

    def se_local():
        gt, ut = lin(xt, wgt), lin(xt, wut)
        a = ttnn.situ_glu(gt, ut, d.beta, d.linear_beta)
        ttnn.deallocate(gt)
        ttnn.deallocate(ut)
        y = lin(a, wdt)
        ttnn.deallocate(a)
        return y

    y = timed(f"local dense SE [{S}, I {I}]", se_local)
    rel, p = _err(own_rows(y, S), yref)
    logger.info(f"SETP local dense output vs fp32: rel {rel * 100:.2f}%  pcc {p:.5f}")
