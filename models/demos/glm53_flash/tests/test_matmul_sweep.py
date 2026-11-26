# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode projection matmul config sweep (M = 32, bfp8 weights, one chip): for every GLM-5.3-Flash per-chip
projection shape, time `ttnn.linear` default vs 1D-mcast (in0 multicast, interleaved in/out) over the
core-count / per_core_N / in0_block_w space vs DRAM-sharded weights INCLUDING the activation reshard, and
print the best config per shape (JSON line `BEST {...}`).

    TT_VISIBLE_DEVICES=4 pytest models/demos/glm53_flash/tests/test_matmul_sweep.py -s
"""

from __future__ import annotations

import json
import math
import time

import pytest
import torch
from loguru import logger

import ttnn

SHAPES = [  # (name, K, N)
    ("dsa_q_a", 4096, 1536),
    ("dsa_kv_a", 4096, 512),
    ("dsa_q_b", 1536, 4096),
    ("dsa_o_proj", 4096, 4096),
    ("kda_qkv", 4096, 6144),
    ("kda_o_proj", 2048, 4096),
    ("kda_f_a", 4096, 512),
    ("kda_f_b", 512, 2048),
    ("kda_beta", 4096, 32),
    ("kda_g_a", 4096, 512),
    ("kda_g_b", 512, 2048),
    ("shared_w13", 4096, 1024),
    ("shared_w2", 512, 4096),
    ("dense_w13", 4096, 6144),
    ("dense_w2", 3072, 4096),
    ("router_gate", 4096, 288),
]
M = 32
N_REP = 40


def _divisors(n, cap=16):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def _timed(device, fn, n_rep=N_REP):
    out = fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / n_rep * 1e6
    ttnn.release_trace(device, tid)
    return out, us


def _grid_for(blocks, gx, gy):
    """Smallest rectangle (cx <= gx, cy <= gy) holding `blocks` cores, or None."""
    if blocks > gx * gy:
        return None
    cx = min(gx, blocks)
    cy = math.ceil(blocks / cx)
    return (cx, cy)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000, "l1_small_size": 32768}], indirect=True)
def test_decode_matmul_sweep(mesh_device):
    device = mesh_device
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    dram_cores = device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    hifi2 = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    logger.info(f"compute grid {gx}x{gy}, dram cores {dram_cores}")
    torch.manual_seed(0)
    bests = {}
    for name, K, N in SHAPES:
        Kt, Nt = K // 32, math.ceil(N / 32)
        w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        ref = (x.float() @ w.float())[0, 0]
        x_i = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_i = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        rows = []

        def check(out):
            o = ttnn.to_torch(out).float()
            o = o.reshape(-1, o.shape[-1])[:M, :N]
            return torch.corrcoef(torch.stack([ref.flatten(), o.flatten()]))[0, 1].item()

        out, us = _timed(device, lambda: ttnn.linear(x_i, w_i, dtype=ttnn.bfloat16, compute_kernel_config=hifi2))
        rows.append(("default", us, check(out), {}))
        # 1D mcast_in0 sweep
        for pcn in (1, 2, 3, 4, 6, 8):
            blocks = math.ceil(Nt / pcn)
            g = _grid_for(blocks, gx, gy)
            if g is None:
                continue
            cx, cy = g
            for ibw in _divisors(Kt, 16):
                if ibw * pcn > 64:  # in0 + in1 block tiles (bf16 2 KB, bfp8 1 KB) stay far below L1
                    continue
                sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(cx, cy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=sub_w,
                    out_block_h=1,
                    out_block_w=pcn,
                    per_core_M=1,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                try:
                    out, us = _timed(
                        device,
                        lambda: ttnn.linear(
                            x_i, w_i, program_config=pc, dtype=ttnn.bfloat16, compute_kernel_config=hifi2
                        ),
                    )
                    rows.append(
                        (
                            f"1d {cx}x{cy} pcn={pcn} ibw={ibw}",
                            us,
                            check(out),
                            {"kind": "1d", "grid": [cx, cy], "per_core_N": pcn, "in0_block_w": ibw},
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    rows.append(
                        (
                            f"1d {cx}x{cy} pcn={pcn} ibw={ibw}",
                            float("inf"),
                            float("nan"),
                            {"err": str(e).splitlines()[0][:100]},
                        )
                    )
        # DRAM-sharded (weights width-sharded over the DRAM banks) including the activation reshard in and out
        padded = math.ceil(N / (32 * dram_cores)) * (32 * dram_cores)
        try:
            w_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (K, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
            )
            w_ds = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem)
            act_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
            x_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(act_grid, (M, K // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
            )
            out_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(act_grid, (M, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
            )
            for ibw in _divisors(K // (32 * dram_cores), 16):
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=ibw, per_core_M=1, per_core_N=math.ceil(N / (32 * dram_cores)), fused_activation=None
                )

                def ds():
                    xs = ttnn.to_memory_config(x_i, x_mem)
                    o = ttnn.linear(
                        xs,
                        w_ds,
                        program_config=pc,
                        memory_config=out_mem,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=hifi2,
                    )
                    return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)

                out, us = _timed(device, ds)
                rows.append(
                    (f"dram-sharded ibw={ibw} (+reshards)", us, check(out), {"kind": "dram", "in0_block_w": ibw})
                )
        except Exception as e:  # noqa: BLE001
            rows.append(("dram-sharded", float("inf"), float("nan"), {"err": str(e).splitlines()[0][:100]}))
        rows.sort(key=lambda r: r[1])
        logger.info(f"{name}: [{M} x {K}] x [{K} x {N}]  ({K * N / 1e6:.1f} MB bfp8)")
        for label, us, p, _ in rows[:6]:
            logger.info(f"  {label:40s} {us:8.1f} us  {K * N / us / 1e3:7.1f} GB/s  pcc {p:.5f}")
        d = next(r for r in rows if r[0] == "default")
        logger.info(
            f"  {'default':40s} {d[1]:8.1f} us  {K * N / d[1] / 1e3:7.1f} GB/s  (speedup best {d[1] / rows[0][1]:.2f}x)"
        )
        ok = [r for r in rows if r[2] == r[2] and r[2] > 0.9999]
        bests[name] = {"K": K, "N": N, "default_us": d[1], "best_us": ok[0][1], "best": ok[0][3], "label": ok[0][0]}
        ttnn.deallocate(x_i)
        ttnn.deallocate(w_i)
    for name, b in bests.items():
        logger.info("BEST " + json.dumps({"name": name, **b}))
    tot_def = sum(b["default_us"] for b in bests.values())
    tot_best = sum(b["best_us"] for b in bests.values())
    logger.info(f"sum over shapes: default {tot_def:.0f} us -> best {tot_best:.0f} us")
