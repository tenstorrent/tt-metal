# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA absorb batched matmuls at decode shape ([1, 16, 32, Dq] x [1, 16, Dq, L] and [1, 16, 32, L] x [1, 16, L, Dv]):
default dispatch vs MatmulMultiCoreReuseProgramConfig (one output block per core) over per_core_N / in0_block_w.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_bmm_sweep.py -s
"""

from __future__ import annotations

import json
import math
import time

import pytest
import torch
from loguru import logger

import ttnn

SHAPES = [("absorb_q", 16, 32, 256, 512), ("absorb_o", 16, 32, 512, 256)]  # (name, H, M, K, N)
N_REP = 40


def _timed(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    ttnn.release_trace(device, tid)
    return out, us


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000, "l1_small_size": 32768}], indirect=True)
def test_bmm_sweep(mesh_device):
    device = mesh_device
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    hifi2 = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    for name, H, M, K, N in SHAPES:
        Kt, Nt = K // 32, N // 32
        a = torch.randn(1, H, M, K, dtype=torch.bfloat16)
        w = torch.randn(1, H, K, N, dtype=torch.bfloat16) * 0.05
        ref = a.float() @ w.float()
        a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_t = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        def check(o):
            return torch.corrcoef(torch.stack([ref.flatten(), ttnn.to_torch(o).float().flatten()]))[0, 1].item()

        rows = []
        o, us = _timed(device, lambda: ttnn.matmul(a_t, w_t, dtype=ttnn.bfloat16, compute_kernel_config=hifi2))
        rows.append(("default", us, check(o)))
        for pcn in (1, 2, 4, 8, 16):
            if Nt % pcn:
                continue
            blocks = H * (Nt // pcn)
            if blocks > gx * gy:
                continue
            cx = min(gx, blocks)
            cy = math.ceil(blocks / cx)
            for ibw in (d for d in (1, 2, 4, 8, 16) if Kt % d == 0):
                sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(cx, cy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=sub_w,
                    per_core_M=1,
                    per_core_N=pcn,
                )
                try:
                    o, us = _timed(
                        device,
                        lambda: ttnn.matmul(
                            a_t, w_t, program_config=pc, dtype=ttnn.bfloat16, compute_kernel_config=hifi2
                        ),
                    )
                    rows.append((f"bmm {cx}x{cy} pcn={pcn} ibw={ibw}", us, check(o)))
                except Exception as e:  # noqa: BLE001
                    rows.append((f"bmm {cx}x{cy} pcn={pcn} ibw={ibw}", float("inf"), float("nan")))
                    logger.info(f"  {name} bmm {cx}x{cy} pcn={pcn} ibw={ibw} failed: {str(e).splitlines()[0][:120]}")
        rows.sort(key=lambda r: r[1])
        logger.info(f"{name}: [{H} x {M} x {K}] x [{H} x {K} x {N}]")
        for label, us, p in rows[:6]:
            logger.info(f"  {label:36s} {us:8.1f} us  pcc {p:.5f}")
        d = next(r for r in rows if r[0] == "default")
        logger.info(f"  {'default':36s} {d[1]:8.1f} us  (best {d[1] / rows[0][1]:.2f}x)")
        ok = [r for r in rows if r[2] == r[2] and r[2] > 0.9999]
        logger.info("BEST " + json.dumps({"name": name, "default_us": d[1], "best_us": ok[0][1], "label": ok[0][0]}))
