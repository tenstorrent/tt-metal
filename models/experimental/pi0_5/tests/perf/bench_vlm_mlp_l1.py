# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Faithful L1-vs-DRAM weight microbench for the VLM Gemma-2B MLP matmuls.

The original task hypothesis was L1 weight residency. I PROVED SigLIP matmuls
are math-bound (L1 vs DRAM = 1.00x, no benefit). The open question: are the
much larger VLM MLP matmuls (gate/up 2048->16384, down 16384->2048) bandwidth-
bound instead — where L1 residency WOULD help?

Same method that characterized SigLIP: time one matmul with weights in DRAM vs
L1 (interleaved), at the VLM shape. The relative delta is hardware-independent
(works on this harvested chip). If L1 >> DRAM -> bandwidth-bound -> L1 residency
is the right lever for the VLM (unlike SigLIP).

Run:
    source models/experimental/pi0_5/local_env.sh
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_vlm_mlp_l1.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import ttnn

WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))
ITERS = int(os.environ.get("BENCH_ITERS", "40"))
WIDTH = 2048
MLP = 16384
SEQ = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "768"))  # bs=2 single-pass


def _time(fn, sync):
    ts = []
    for i in range(WARMUP + ITERS):
        ttnn.synchronize_device(sync)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(sync)
        if i >= WARMUP:
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts)


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        g = dev.compute_with_storage_grid_size()
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        torch.manual_seed(0)
        M = SEQ
        print(f"\n=== VLM MLP L1-vs-DRAM weight microbench  M={M} grid={g.x}x{g.y} LoFi ===")
        print(f"{'shape':12s} {'K':6s} {'N':6s} | {'dram_w':>9s} {'l1_w':>9s} | {'l1_speedup':>10s}")
        # gate/up: 2048->16384 ; down: 16384->2048
        for name, K, N in [("gate/up", WIDTH, MLP), ("down", MLP, WIDTH)]:
            act = ttnn.from_torch(
                torch.randn(1, 1, M, K),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w = torch.randn(K, N)
            w_dram = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            # L1 weight: only attempt if it fits (16384*2048 bf8 = ~34MB won't fit;
            # gate/up = 2048*16384 same). Guard.
            w_bytes = K * N * 17 // 16
            l1s = "n/a (>L1)"
            if w_bytes < 100_000_000:  # let ttnn try; catch OOM
                try:
                    w_l1 = ttnn.from_torch(
                        w,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )

                    def mm_l1():
                        return ttnn.matmul(act, w_l1, compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG)

                    t_l1 = _time(mm_l1, dev)
                    l1s = f"{t_l1:.4f}"
                    ttnn.deallocate(w_l1)
                except Exception as e:
                    l1s = f"OOM/{repr(e)[:18]}"

            def mm_dram():
                return ttnn.matmul(act, w_dram, compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG)

            try:
                t_dram = _time(mm_dram, dev)
                drs = f"{t_dram:.4f}"
                sp = (t_dram / t_l1) if (l1s not in ("n/a (>L1)",) and not l1s.startswith("OOM")) else 0.0
            except Exception as e:
                drs = f"ERR/{repr(e)[:18]}"
                sp = 0.0
            print(f"{name:12s} {K:6d} {N:6d} | {drs:>9s} {l1s:>9s} | {sp:9.2f}x")
            ttnn.deallocate(act)
            ttnn.deallocate(w_dram)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
