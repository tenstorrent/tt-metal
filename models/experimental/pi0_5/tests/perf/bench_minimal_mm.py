# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbench ttnn.experimental.minimal_matmul vs the BS ttnn.linear for the
SigLIP MLP fc1/fc2 shapes. minimal_matmul is the prefill-fast matmul the VLM
MLP uses (LoFi + fused GELU). Question: does it beat the block-sharded linear
for SigLIP's fc1 (1152->4608, GELU) and fc2 (4608->1152)?

minimal_matmul needs TILE interleaved inputs; the BS linear consumes block-
sharded. We measure the matmul itself (interleaved in for minimal; BS in for
linear) at SigLIP M=768.

Run:
    source models/experimental/pi0_5/local_env.sh
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_minimal_mm.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_siglip import _SIGLIP_BS_GRID, _build_bs_matmul_pcfg, _make_bs_memcfg

WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
HIDDEN = 1152
INTER_PAD = 4608
M = 768  # bs=3 * seq=256


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
        gx, gy = _SIGLIP_BS_GRID
        ck_lofi = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        ck_bs = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        torch.manual_seed(0)
        print(f"\n=== minimal_matmul vs BS linear, M={M}, LoFi ===")
        # fc1: K=1152 N=4608 GELU ; fc2: K=4608 N=1152
        for name, K, N, act in [("fc1", HIDDEN, INTER_PAD, "gelu"), ("fc2", INTER_PAD, HIDDEN, None)]:
            w = torch.randn(K, N)
            w_t = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            act_il = ttnn.from_torch(
                torch.randn(1, 1, M, K),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # --- minimal_matmul (interleaved in) ---
            mcfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=4,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            fa = (ttnn.UnaryOpType.GELU, True) if act == "gelu" else None

            def _minimal():
                return ttnn.experimental.minimal_matmul(
                    act_il,
                    w_t,
                    fused_activation=fa,
                    config=mcfg,
                    compute_kernel_config=ck_lofi,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                )

            try:
                t_min = _time(_minimal, dev)
                mins = f"{t_min:.4f}"
            except Exception as e:
                mins = f"ERR {repr(e)[:50]}"

            # --- BS linear (block-sharded in) ---
            bs_hidden = _make_bs_memcfg(1, M, K, gx, gy)
            bs_out = _make_bs_memcfg(1, M, N, gx, gy)
            act_bs = ttnn.to_memory_config(act_il, bs_hidden)
            k_t, n_t, m_t = K // 32, N // 32, M // 32
            bs_act = (ttnn.UnaryOpType.GELU, True) if act == "gelu" else None
            pcfg = _build_bs_matmul_pcfg(m_t, k_t, n_t, gx, gy, activation=bs_act, dst_budget=4)

            def _bslin():
                return ttnn.linear(
                    act_bs,
                    w_t,
                    dtype=ttnn.bfloat8_b,
                    memory_config=bs_out,
                    compute_kernel_config=ck_bs,
                    program_config=pcfg,
                )

            try:
                t_bs = _time(_bslin, dev)
                bss = f"{t_bs:.4f}"
            except Exception as e:
                bss = f"ERR {repr(e)[:50]}"

            print(f"  {name}: minimal={mins}ms  BS_linear={bss}ms")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
