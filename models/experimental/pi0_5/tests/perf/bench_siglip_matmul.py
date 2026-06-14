# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbench one SigLIP-shaped matmul under different weight placements to
find whether the encoder matmuls are math-bound or weight-bandwidth-bound.

For each of the 4 SigLIP block matmuls (qkv, o, fc1, fc2) we time:
  - weights interleaved DRAM (current path)
  - weights interleaved L1 (best case, ignores residency budget)
  - weights DRAM width-sharded + DRAM-sharded matmul pcfg (the micro-op lever)

If L1 >> DRAM the matmul is weight-bandwidth-bound and a streaming/sharded
weight path will help. If they're equal it's math-bound and weight placement
is irrelevant.

Run:
    source models/experimental/pi0_5/local_env.sh
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_matmul.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_common import build_dram_width_sharded_memcfg

WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("BENCH_ITERS", "30"))

HIDDEN = 1152
INTER_PAD = 4608
BS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
SEQ = 256
M = BS * SEQ  # 768

# (name, K, N)
SHAPES = [
    ("qkv", HIDDEN, 3 * HIDDEN),  # 1152 -> 3456
    ("o", HIDDEN, HIDDEN),  # 1152 -> 1152
    ("fc1", HIDDEN, INTER_PAD),  # 1152 -> 4608
    ("fc2", INTER_PAD, HIDDEN),  # 4608 -> 1152
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
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        ckcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        torch.manual_seed(0)
        print(f"\n=== SigLIP matmul microbench  M={M} (bs={BS}*seq={SEQ})  warmup={WARMUP} iters={ITERS} ===")
        print(f"{'shape':6s} {'K':5s} {'N':5s} | {'dram_il':>9s} {'l1_il':>9s} {'dram_ws':>9s} | {'l1_speedup':>10s}")
        for name, K, N in SHAPES:
            act = ttnn.from_torch(
                torch.randn(1, M, K),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w = torch.randn(K, N)
            w_dram = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w_l1 = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
            )

            def mm(weight):
                return lambda: ttnn.linear(
                    act, weight, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckcfg
                )

            d_il, _ = _time(mm(w_dram), dev)
            l_il, _ = _time(mm(w_l1), dev)

            # DRAM width-sharded weights + DRAM-sharded matmul pcfg
            dws = "n/a"
            try:
                memcfg, padded_n, dcores = build_dram_width_sharded_memcfg(dev, K, N)
                wpad = w
                if padded_n != N:
                    wpad = torch.nn.functional.pad(w, (0, padded_n - N))
                w_ws = ttnn.from_torch(
                    wpad, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=memcfg
                )
                # activation must be L1 width-sharded for the DRAM-sharded matmul
                act_ws = ttnn.to_memory_config(
                    act,
                    ttnn.create_sharded_memory_config(
                        (1, 1, M, K),
                        core_grid=ttnn.CoreGrid(y=1, x=dcores),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                pcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=K // 32 // dcores if (K // 32) % dcores == 0 else 1,
                    per_core_M=M // 32,
                    per_core_N=padded_n // 32 // dcores,
                    fused_activation=None,
                )

                def mm_ws():
                    return ttnn.linear(
                        act_ws,
                        w_ws,
                        dtype=ttnn.bfloat8_b,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=ckcfg,
                        program_config=pcfg,
                    )

                dws_v, _ = _time(mm_ws, dev)
                dws = f"{dws_v:9.4f}"
            except Exception as e:
                dws = f"ERR:{repr(e)[:30]}"

            sp = d_il / l_il if l_il > 0 else 0
            print(f"{name:6s} {K:5d} {N:5d} | {d_il:9.4f} {l_il:9.4f} {dws:>9s} | {sp:9.2f}x")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
