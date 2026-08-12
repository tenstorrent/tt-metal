# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: is the default program config leaving prefill matmul throughput behind?

The functional/fused prefill profile marks the three MLP matmuls and the o_proj
matmul ``SLOW`` (~95 TFLOPs, ~30 GB/s) while the QKV matmul reaches 206 TFLOPs
on the same dtype and fidelity.  This measures the auto-selected config against
explicit ``MatmulMultiCoreReuseMultiCastProgramConfig`` tilings (and
``ttnn.experimental.minimal_matmul``) at the real prefill shapes, so the fusing
stage can either take the win or hand the geometry sweep to the optimized stage
with numbers attached.
"""

from __future__ import annotations

import math
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

SHAPES = [
    ("wqkv   ", 8192, 6656, 4608),
    ("o_proj ", 8192, 4096, 6656),
    ("mlp_gate", 8192, 6656, 19968),
    ("mlp_down", 8192, 19968, 6656),
]
#: Every rectangle worth trying.  K is 6656 or 4096 or 19968, all multiples of
#: 32; the 2D program config needs K/(32*grid_y) to be an integer, so grid
#: heights 1/2/4/8 are the ones that can divide all of them.  Widths span the
#: whole 11-wide Blackhole grid.
GRIDS = [(8, 1), (8, 2), (8, 4), (8, 8), (8, 10), (11, 1), (11, 2), (11, 4), (10, 10), (11, 10)]
ITERS = 3
ROUNDS = 2


def largest_divisor(value, maximum=8):
    for candidate in range(min(maximum, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def out_subblock_w(per_core_n):
    for candidate in range(min(4, per_core_n), 0, -1):
        if per_core_n % candidate == 0:
            return candidate
    return 1


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for label, m, k, n in SHAPES:
            torch.manual_seed(0)
            a = torch.randn(1, 1, m, k).to(torch.bfloat16) * 0.1
            b = torch.randn(1, 1, k, n).to(torch.bfloat16) * 0.02
            ta = ttnn.from_torch(
                a, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tb = ttnn.from_torch(
                b, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ref = a.float() @ b.float()

            def timed(tag, fn):
                try:
                    out = fn()
                except Exception as exc:  # noqa: BLE001
                    print(f"MM {label} {tag:36s} FAILED {type(exc).__name__}: {str(exc)[:130]}", flush=True)
                    return
                pcc = comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]
                ttnn.deallocate(out)
                rounds = []
                for _ in range(ROUNDS):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(mesh)
                    rounds.append((time.perf_counter() - t0) / ITERS * 1e3)
                dt = min(rounds)
                tflops = 2 * m * k * n / (dt * 1e-3) / 1e12
                print(f"MM {label} {tag:36s} {dt:9.3f} ms  {tflops:7.1f} TFLOPs  PCC={pcc}", flush=True)

            timed("auto (shipped)", lambda: ttnn.linear(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            timed(
                "auto+ck", lambda: ttnn.linear(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck)
            )
            for gx, gy in GRIDS:
                per_core_m = math.ceil(m / (32 * gy))
                per_core_n = math.ceil(n / (32 * gx))
                if k % (32 * gy):
                    continue
                in0 = largest_divisor(k // (32 * gy))
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=in0,
                    out_subblock_h=1,
                    out_subblock_w=out_subblock_w(per_core_n),
                    per_core_M=per_core_m,
                    per_core_N=per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                timed(
                    f"2D {gx}x{gy} in0={in0} pcN={per_core_n}",
                    lambda pc=pc: ttnn.linear(
                        ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc, compute_kernel_config=ck
                    ),
                )
            timed("minimal_matmul default ck", lambda: ttnn.experimental.minimal_matmul(ta, tb))
            timed(
                "minimal_matmul hifi2 ck (shipped)",
                lambda: ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck),
            )
            # The pack-time activation rejection was measured on ttnn.linear;
            # minimal_matmul exposes its own fused_activation, so re-test it on
            # the op the shipped prefill actually uses.
            timed(
                "minimal_matmul fused_activation=silu",
                lambda: ttnn.experimental.minimal_matmul(
                    ta, tb, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), compute_kernel_config=ck
                ),
            )
            # MinimalMatmulConfig block sweep (tile units) on the shipped
            # kernel, over the full compute grid.  The shipped path passes no
            # config at all and lets the op pick.
            full_grid = ttnn.CoreCoord(grid.x, grid.y)
            for mb, kb, nb, sh, sw in (
                (4, 2, 4, 1, 4),
                (8, 2, 4, 1, 4),
                (4, 4, 4, 1, 4),
                (8, 4, 8, 1, 4),
                (4, 2, 8, 1, 4),
                (16, 2, 4, 1, 4),
            ):
                timed(
                    f"minimal cfg M{mb} K{kb} N{nb} sw{sw}",
                    lambda mb=mb, kb=kb, nb=nb, sh=sh, sw=sw: ttnn.experimental.minimal_matmul(
                        ta,
                        tb,
                        compute_kernel_config=ck,
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=mb,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sh,
                            subblock_w=sw,
                            compute_with_storage_grid_size=full_grid,
                        ),
                    ),
                )
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
