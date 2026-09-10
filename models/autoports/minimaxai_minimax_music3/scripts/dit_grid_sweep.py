#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Targeted DiT matmul sweep (stage 07, second round): the op default already runs the block matmuls on the full
11x10 grid with padded per-core blocks but ``in0_block_w = 1``; try explicit full-grid configs with larger K blocks
and fp16 accumulation (subblocks up to 8 tiles), a fused SiLU, and ``silu(g) * a`` fused into the multiply.

    with_hw_lock timeout 1200 $MM3_PY $MM3_MODEL_DIR/scripts/dit_grid_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "optimize" / "sweeps"


def time_op(dev, fn, ops=6, reps=5):
    fn()
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


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000)
    dev.enable_program_cache()
    rows = []
    try:
        grid = dev.compute_with_storage_grid_size()
        gx, gy = grid.x, grid.y
        cc16 = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        g = torch.Generator().manual_seed(0)
        for m, k, n in ((1536, 2048, 8192), (1536, 8192, 2048), (1536, 2048, 6144), (1536, 2048, 2048)):
            a = ttnn.from_torch(
                torch.randn(1, 1, m, k, generator=g).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )
            w = ttnn.from_torch(
                torch.randn(1, 1, k, n, generator=g) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev
            )
            mt, kt, nt = m // 32, k // 32, n // 32
            per_m, per_n = -(-mt // gy), -(-nt // gx)
            variants = [("default_fp16acc", None)]
            for blk in (1, 2, 4, 8):
                if kt % blk:
                    continue
                for sh, sw in ((1, 8), (1, 6), (1, 4), (2, 4), (2, 3), (1, 3), (1, 2)):
                    if per_m % sh or per_n % sw or sh * sw > 8:
                        continue
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
                    variants.append((f"2d_g{gx}x{gy}_k{blk}_s{sh}x{sw}_fp16acc", pc))
                    break
            ref = None
            for name, pc in variants:
                try:
                    dt, out = time_op(
                        dev,
                        lambda: ttnn.matmul(
                            a,
                            w,
                            program_config=pc,
                            compute_kernel_config=cc16,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ),
                    )
                    if ref is None:
                        ref = out
                    pcc = float(comp_pcc(ref, out, 0.0)[1])
                    rows.append({"shape": [m, k, n], "variant": name, "us": dt * 1e6, "pcc_vs_default": pcc})
                    logger.info(f"[{m}x{k}x{n}] {name:34s} {dt*1e6:8.1f} us  pcc {pcc:.6f}")
                except Exception as exc:  # noqa: BLE001
                    rows.append({"shape": [m, k, n], "variant": name, "error": repr(exc)[:200]})
                    logger.warning(f"[{m}x{k}x{n}] {name}: {repr(exc)[:160]}")
            ttnn.deallocate(a)
            ttnn.deallocate(w)
        # silu fused into the gate multiply
        a = ttnn.from_torch(
            torch.randn(1, 1, 1536, 8192, generator=g).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        gt = ttnn.from_torch(
            torch.randn(1, 1, 1536, 8192, generator=g).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        dt, ref = time_op(dev, lambda: ttnn.multiply(a, ttnn.silu(gt), memory_config=ttnn.DRAM_MEMORY_CONFIG))
        rows.append({"shape": "silu+mul", "variant": "separate", "us": dt * 1e6})
        logger.info(f"silu then multiply: {dt*1e6:.1f} us")
        for name, fn in (
            (
                "mul_input_b_silu",
                lambda: ttnn.multiply(
                    a, gt, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
            ),
            (
                "mul_b_silu_enum",
                lambda: ttnn.multiply(
                    a,
                    gt,
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            ),
        ):
            try:
                dt, out = time_op(dev, fn)
                pcc = float(comp_pcc(ref, out, 0.0)[1])
                rows.append(
                    {
                        "shape": "silu+mul",
                        "variant": name,
                        "us": dt * 1e6,
                        "pcc_vs_default": pcc,
                        "max_abs": float((ref - out).abs().max()),
                    }
                )
                logger.info(f"{name}: {dt*1e6:.1f} us pcc {pcc:.6f}")
            except Exception as exc:  # noqa: BLE001
                rows.append({"shape": "silu+mul", "variant": name, "error": repr(exc)[:200]})
                logger.warning(f"{name}: {repr(exc)[:200]}")
    finally:
        ttnn.close_mesh_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dit_grid.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
