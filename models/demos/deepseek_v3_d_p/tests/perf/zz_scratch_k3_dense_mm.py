# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe: matmul program configs for the K3 block's dense ops at 320 rows per chip (device time, slowest
chip, vs the DRAM floor of the weight). Shapes: latent down / up, shared gate / up / down, router."""
import os

import pytest
import torch
from loguru import logger

import models.demos.glm53_flash_d_p.tt.mm_configs as mmc
import ttnn
from models.demos.deepseek_v3.utils.config_helpers import dram_sharded_weight_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

M = 320
SHAPES = {  # name: (K, N, weight dtype)
    "lat_down": (7168, 3584, ttnn.bfloat8_b),
    "lat_up": (3584, 7168, ttnn.bfloat8_b),
    "se_gu": (7168, 6144, ttnn.bfloat8_b),
    "se_down": (6144, 7168, ttnn.bfloat8_b),
    "router": (7168, int(os.environ.get("MM_E", "448")), ttnn.bfloat16),
}
_PARAMS = [pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="2x4")]
_KEY = "DEVICE KERNEL DURATION [ns]"


def _times(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    rows = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        rows.append([float(p.program_analyses_results[_KEY].duration) / 1e3 for p in programs if _KEY in (p.program_analyses_results or {})])
    return rows


def _configs(Kt, Nt, gx, gy):
    """(label, program_config) candidates."""
    out = [("default", None)]
    n = gx * gy
    for bw in (2, 4, 8):
        if Kt % bw:
            continue
        pn = -(-Nt // n)
        sw = max(d for d in (1, 2, 4, 8) if pn % d == 0 and d <= 8)
        out.append((f"1d_in0 bw{bw} pn{pn}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=1, out_subblock_w=sw,
            out_block_h=10, out_block_w=pn, per_core_M=10, per_core_N=pn, fuse_batch=True, fused_activation=None,
            mcast_in0=True)))
    for gyy in (5, 10):
        pm = -(-10 // gyy)
        pn = -(-Nt // gx)
        for bw in (2, 4, 8):
            if Kt % bw:
                continue
            sw = max(d for d in (1, 2, 4, 8) if pn % d == 0 and d * pm <= 8)
            out.append((f"2d {gx}x{gyy} bw{bw} pm{pm} pn{pn}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gyy), in0_block_w=bw, out_subblock_h=1,
                out_subblock_w=sw, out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                fused_activation=None, fuse_batch=True)))
    return out


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_dense_mm(mesh_device, device_params):
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    names = os.environ.get("MM_SHAPES", ",".join(SHAPES)).split(",")
    fids = [ttnn.MathFidelity[f] for f in os.environ.get("MM_FID", "HiFi2,LoFi").split(",")]
    for name in names:
        Kd, Nd, wdt = SHAPES[name]
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wts = {"il": wt}
        try:
            wts["ds"] = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep,
                                        memory_config=dram_sharded_weight_config(Kd, Nd, mesh_device.dram_grid_size()))
        except Exception as e:  # noqa: BLE001
            logger.info(f"MM {name} dram-sharded weight FAILED {str(e).splitlines()[0][:150]}")
        mmc.ENABLED = True
        mm2d = mmc.mm2d(mesh_device, M, Kd, Nd, ttnn.bfloat16, wdt, ttnn.bfloat16)
        ref = x.float() @ w.float()
        wbytes = Kd * Nd * (1.0625 if wdt == ttnn.bfloat8_b else 2)
        floor = wbytes / 512e3
        best = None
        for fid in fids:
            cfg = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            cands = [(f"il {l}", pc, "il") for l, pc in _configs(Kd // 32, Nd // 32, g.x, g.y)]
            cands += [("il mm2d", mm2d, "il")] + ([("ds mm2d", mm2d, "ds")] if "ds" in wts else [])
            for label, pc, wk in cands:
                try:
                    _times(mesh_device)
                    us = []
                    for _ in range(3):
                        y = ttnn.linear(xt, wts[wk], compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc)
                        ttnn.synchronize_device(mesh_device)
                        us.append(max(r[-1] for r in _times(mesh_device)))
                        yt = y
                    got = ttnn.to_torch(ttnn.get_device_tensors(yt)[0]).float()
                    p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                    ttnn.deallocate(yt)
                    t = min(us)
                    logger.info(f"MM {name:8s} {str(fid).split('.')[-1]:5s} {label:28s} {t:8.1f} us  ({floor / t * 100:5.1f}% of DRAM floor {floor:.1f})  pcc {p:.5f}")
                    if best is None or t < best[0]:
                        best = (t, str(fid), label)
                except Exception as e:  # noqa: BLE001
                    logger.info(f"MM {name:8s} {str(fid).split('.')[-1]:5s} {label:28s} FAILED {str(e).splitlines()[0][:150]}")
        logger.info(f"MM_BEST {name}: {best}  floor {floor:.1f}")
        ttnn.deallocate(xt)
        for t in wts.values():
            ttnn.deallocate(t)
