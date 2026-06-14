# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Faithful SigLIP block-sharded matmul microbench: L1 vs DRAM weights, using
the EXACT 12x8 BS grid + program config the encoder uses (forward_bs path).

This corrects bench_siglip_matmul.py, which used the default pcfg (grid-
starved) and so measured launch overhead, not the real matmul.

Run:
    source models/experimental/pi0_5/local_env.sh
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_matmul2.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_siglip import (
    _SIGLIP_BS_GRID,
    _build_bs_matmul_pcfg,
    _make_bs_memcfg,
)

WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("BENCH_ITERS", "50"))

HIDDEN = 1152
INTER_PAD = 4608
BS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
SEQ = 256
M = BS * SEQ  # 768

# (name, K, N, activation)
SHAPES = [
    ("qkv", HIDDEN, 3 * HIDDEN, None),
    ("o", HIDDEN, HIDDEN, None),
    ("fc1", HIDDEN, INTER_PAD, "gelu"),
    ("fc2", INTER_PAD, HIDDEN, None),
]


def _time(fn, sync):
    ts = []
    for i in range(WARMUP + ITERS):
        ttnn.synchronize_device(sync)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(sync)
        if i >= WARMUP:
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        gx, gy = _SIGLIP_BS_GRID  # 12, 8
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        ck_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        torch.manual_seed(0)
        m_tiles = M // 32
        print(f"\n=== Faithful SigLIP BS matmul microbench  M={M} grid={gx}x{gy}  warmup={WARMUP} iters={ITERS} ===")
        print(f"{'shape':6s} {'K':5s} {'N':6s} | {'HiFi2':>9s} {'LoFi':>9s} | {'speedup':>8s}")
        total_d = total_l = 0.0
        for name, K, N, act in SHAPES:
            k_t, n_t = K // 32, N // 32
            in_memcfg = _make_bs_memcfg(1, 1, M, K, gx, gy) if False else _make_bs_memcfg(BS, M, K, gx, gy)
            # activation block-sharded on the grid: shape (b=1,1,M,K) flattened M
            act_t = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
            act_bs = ttnn.to_memory_config(
                act_t,
                ttnn.create_sharded_memory_config(
                    (1, 1, M, K),
                    core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            out_memcfg = ttnn.create_sharded_memory_config(
                (1, 1, M, N),
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            pcfg = _build_bs_matmul_pcfg(m_tiles, k_t, n_t, gx, gy, activation=act, dst_budget=4)
            w = torch.randn(K, N)
            w_dram = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w_l1 = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
            )

            def mm(weight, kc):
                return lambda: ttnn.linear(
                    act_bs,
                    weight,
                    dtype=ttnn.bfloat8_b,
                    memory_config=out_memcfg,
                    compute_kernel_config=kc,
                    program_config=pcfg,
                )

            try:
                d, _ = _time(mm(w_dram, ck), dev)
                l, _ = _time(mm(w_dram, ck_lofi), dev)
                total_d += d
                total_l += l
                sp = d / l if l > 0 else 0
                print(f"{name:6s} {K:5d} {N:6d} | {d:9.4f} {l:9.4f} | {sp:7.2f}x")
            except Exception as e:
                print(f"{name:6s} {K:5d} {N:6d} | ERR {repr(e)[:60]}")
            ttnn.deallocate(w_dram)
            ttnn.deallocate(w_l1)
            ttnn.deallocate(act_bs)
        if total_l > 0:
            print(f"{'TOTAL':6s} {'':5s} {'':6s} | {total_d:9.4f} {total_l:9.4f} | {total_d/total_l:7.2f}x")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
