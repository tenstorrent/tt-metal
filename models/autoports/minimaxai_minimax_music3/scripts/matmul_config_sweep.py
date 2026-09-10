#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config / fidelity micro-sweep for the stage-07 hot matmuls and the DiT SDPA (device time via trace replay).

Groups (``--group``): ``depth`` (the depth decoder's [64 x 4096 x {12288, 6144, 4096}] and [64 x 6144 x 4096] bfp8
matmuls), ``dit`` (the DiT's [1536 x 2048 x {8192, 6144, 2048}] and [1536 x 8192 x 2048] bfp8 matmuls, bf16
activations) and ``sdpa`` (the DiT attention, B 2 x 32 heads x S 768 x 64, chunk sizes). Every candidate is timed as
the mean of ``--reps`` replays of a trace holding ``--ops`` back-to-back copies of the op, after a compile run, and
its output is compared (PCC) with the default HiFi2 program. Results: ``doc/optimize/sweeps/<group>.json``.

    with_hw_lock timeout 1800 $MM3_PY $MM3_MODEL_DIR/scripts/matmul_config_sweep.py --group depth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "optimize" / "sweeps"


def cfg(dev, fid, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
    )


def time_op(dev, fn, ops, reps):
    fn()  # compile
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(ops)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / (reps * ops)
    out = ttnn.to_torch(outs[0]).float()
    ttnn.release_trace(dev, tid)
    for o in outs:
        ttnn.deallocate(o)
    return dt, out


def mm_candidates(m, k, n, grid_max=(11, 10), max_sub=4):
    """Explicit 2D-mcast program configs for an [m x k x n] matmul: rectangular grids dividing M and N tiles.
    ``max_sub``: tiles per output subblock (4 with fp32 accumulation, 8 with fp16)."""
    mt, kt, nt = m // 32, k // 32, n // 32
    cands = []
    for gy in range(min(grid_max[1], mt), 0, -1):
        if mt % gy:
            continue
        for gx in range(min(grid_max[0], nt), 0, -1):
            if nt % gx:
                continue
            if gx * gy < 32:
                continue
            per_m, per_n = mt // gy, nt // gx
            for blk in (2, 4, 8):
                if kt % blk:
                    continue
                for sh, sw in ((2, 4), (1, 8), (4, 2), (2, 2), (1, 4), (4, 1), (1, 2), (2, 1), (1, 1)):
                    if per_m % sh or per_n % sw or sh * sw > max_sub:
                        continue
                    cands.append((gx, gy, per_m, per_n, blk, sh, sw))
                    break  # largest legal subblock per (grid, blk)
            break  # largest gx per gy
        if len(cands) >= 12:
            break
    return cands


def run_matmul_group(dev, shapes, act_dtype, w_dtype, ops, reps, one_d_variants=False):
    rows = []
    g = torch.Generator().manual_seed(0)
    for m, k, n in shapes:
        a = ttnn.from_torch(
            (torch.randn(1, 1, m, k, generator=g)).to(torch.bfloat16),
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        w = ttnn.from_torch(
            (torch.randn(1, 1, k, n, generator=g) * 0.02), dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=dev
        )
        ref = None
        variants = [
            ("default", None, ttnn.MathFidelity.HiFi2, True),
            ("default_lofi", None, ttnn.MathFidelity.LoFi, True),
            ("default_fp16acc", None, ttnn.MathFidelity.HiFi2, False),
        ]
        for fp32 in (True, False):
            for gx, gy, per_m, per_n, blk, sh, sw in mm_candidates(m, k, n, max_sub=4 if fp32 else 8):
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=blk,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=per_m,
                    per_core_N=per_n,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                variants.append(
                    (f"2d_g{gx}x{gy}_k{blk}_s{sh}x{sw}{'' if fp32 else '_fp16acc'}", pc, ttnn.MathFidelity.HiFi2, fp32)
                )
                if fp32 and m >= 1024:
                    variants.append((f"2d_g{gx}x{gy}_k{blk}_s{sh}x{sw}_lofi", pc, ttnn.MathFidelity.LoFi, True))
        if one_d_variants:
            mt, kt, nt = m // 32, k // 32, n // 32
            for cores, (gx, gy) in ((100, (10, 10)), (64, (8, 8)), (110, (11, 10))):
                if nt % cores and cores != 100:
                    continue
                per_n = -(-nt // cores)
                for blk in (2, 4, 8, 16, 32):
                    if kt % blk:
                        continue
                    for fp32 in (True, False):
                        cap = 4 if fp32 else 8
                        sw = max(w for w in (1, 2, 4, 8) if per_n % w == 0 and w <= per_n)
                        sh = mt if mt * sw <= cap else 1
                        while sh * sw > cap:
                            sw //= 2
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=blk,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            per_core_M=mt,
                            per_core_N=per_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        )
                        tag = f"1d_c{cores}_k{blk}_s{sh}x{sw}{'' if fp32 else '_fp16acc'}"
                        variants.append((tag, pc, ttnn.MathFidelity.HiFi2, fp32))
                        if fp32:
                            variants.append((tag + "_lofi", pc, ttnn.MathFidelity.LoFi, True))
        for name, pc, fid, fp32 in variants:
            try:
                dt, out = time_op(
                    dev,
                    lambda: ttnn.matmul(
                        a,
                        w,
                        program_config=pc,
                        compute_kernel_config=cfg(dev, fid, fp32),
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ops,
                    reps,
                )
                if ref is None:
                    ref = out
                pcc = float(comp_pcc(ref, out, 0.0)[1])
                gbps = (k * n * (1.0625 if w_dtype == ttnn.bfloat8_b else 2.0)) / dt / 1e9
                row = {"shape": [m, k, n], "variant": name, "us": dt * 1e6, "weight_gbps": gbps, "pcc_vs_default": pcc}
                logger.info(f"[{m}x{k}x{n}] {name:28s} {dt*1e6:8.1f} us  {gbps:6.0f} GB/s  pcc {pcc:.6f}")
            except Exception as exc:  # noqa: BLE001
                row = {"shape": [m, k, n], "variant": name, "error": repr(exc)[:200]}
                logger.warning(f"[{m}x{k}x{n}] {name}: {repr(exc)[:160]}")
            rows.append(row)
        ttnn.deallocate(a)
        ttnn.deallocate(w)
    return rows


def run_sdpa(dev, ops, reps):
    rows = []
    B, H, S, D = 2, 32, 768, 64
    g = torch.Generator().manual_seed(0)
    q = ttnn.from_torch(
        torch.randn(B, H, S, D, generator=g).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
    )
    k = ttnn.from_torch(
        torch.randn(B, H, S, D, generator=g).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
    )
    v = ttnn.from_torch(
        torch.randn(B, H, S, D, generator=g).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
    )
    m = torch.zeros(1, 1, S, S)
    m[..., 690:] = -1e9
    mask = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    grid = dev.compute_with_storage_grid_size()
    ref = None
    for qc, kc in ((128, 128), (256, 256), (128, 256), (256, 128), (384, 384), (768, 768), (64, 128), (256, 768)):
        for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
            for use_mask in (True,):
                pc = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    q_chunk_size=qc,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                )
                name = f"q{qc}_k{kc}_{'hifi4' if fid == ttnn.MathFidelity.HiFi4 else 'hifi2'}"
                try:
                    dt, out = time_op(
                        dev,
                        lambda: ttnn.transformer.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=mask if use_mask else None,
                            is_causal=False,
                            program_config=pc,
                            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                                dev.arch(),
                                math_fidelity=fid,
                                math_approx_mode=False,
                                fp32_dest_acc_en=False,
                                packer_l1_acc=False,
                            ),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ),
                        ops,
                        reps,
                    )
                    if ref is None:
                        ref = out
                    pcc = float(comp_pcc(ref[:, :, :690], out[:, :, :690], 0.0)[1])
                    rows.append({"variant": name, "us": dt * 1e6, "pcc_vs_first": pcc})
                    logger.info(f"sdpa {name:22s} {dt*1e6:8.1f} us  pcc {pcc:.6f}")
                except Exception as exc:  # noqa: BLE001
                    rows.append({"variant": name, "error": repr(exc)[:200]})
                    logger.warning(f"sdpa {name}: {repr(exc)[:160]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["depth", "depth12k", "dit", "sdpa", "llm"])
    ap.add_argument("--ops", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000)
    dev.enable_program_cache()
    try:
        if a.group == "depth12k":
            rows = run_matmul_group(
                dev,
                [(64, 4096, 12288), (64, 4096, 7168), (32, 4096, 4096)],
                ttnn.bfloat16,
                ttnn.bfloat8_b,
                a.ops,
                a.reps,
                one_d_variants=True,
            )
        elif a.group == "depth":
            rows = run_matmul_group(
                dev,
                [(64, 4096, 12288), (64, 4096, 6144), (64, 6144, 4096), (64, 4096, 4096)],
                ttnn.bfloat16,
                ttnn.bfloat8_b,
                a.ops,
                a.reps,
                one_d_variants=True,
            )
        elif a.group == "llm":
            rows = run_matmul_group(
                dev,
                [(32, 4096, 12288), (32, 12288, 4096), (32, 4096, 16032)],
                ttnn.bfloat16,
                ttnn.bfloat8_b,
                a.ops,
                a.reps,
                one_d_variants=True,
            )
        elif a.group == "dit":
            rows = run_matmul_group(
                dev,
                [(1536, 2048, 8192), (1536, 8192, 2048), (1536, 2048, 6144), (1536, 2048, 2048)],
                ttnn.bfloat16,
                ttnn.bfloat8_b,
                a.ops,
                a.reps,
            )
        else:
            rows = run_sdpa(dev, a.ops, a.reps)
    finally:
        ttnn.close_mesh_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{a.group}.json"
    p.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info(f"wrote {p}")


if __name__ == "__main__":
    main()
