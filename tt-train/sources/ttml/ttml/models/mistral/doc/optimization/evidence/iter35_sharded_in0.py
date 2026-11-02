# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: the 1D mcast matmul with the activation width-sharded in L1 (every shard core multicasts its K
blocks from L1, no single DRAM-reading sender), gate_up forward shard shape, HiFi4 / fp32 accumulate.

    python iter35_sharded_in0.py --m 1024 --k 5184 --n 16384 --shards 27,54,81
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import ttnn


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / reps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--k", type=int, default=5184)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--shards", default="27,54,81")
    p.add_argument("--out-block-h", default="16,32")
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )
    rng = np.random.default_rng(0)
    a = ttnn.from_torch(
        torch.from_numpy(rng.standard_normal((1, 1, M, K)).astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.from_numpy((rng.standard_normal((1, 1, K, N)) * 0.02).astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    flops = 2 * M * K * N
    cores = grid.x * grid.y
    per_core_n = -(-Nt // cores)

    def pc(w, obh):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=w,
            out_subblock_h=4,
            out_subblock_w=1,
            out_block_h=obh,
            out_block_w=per_core_n,
            per_core_M=Mt,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    ms, ref = bench(lambda: ttnn.matmul(a, b, program_config=pc(2, 16), compute_kernel_config=cfg), args.reps, device)
    ref_h = ttnn.to_torch(ref).float()
    print(f"M={M} K={K} N={N}: interleaved in0, w=2 16x{per_core_n}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOP/s")
    for w in (9, 6, 3):
        if Kt % w == 0:
            ms, out = bench(
                lambda: ttnn.matmul(a, b, program_config=pc(w, 16), compute_kernel_config=cfg), args.reps, device
            )
            print(f"  interleaved in0, w={w} 16x{per_core_n}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOP/s")
    for s in (int(v) for v in args.shards.split(",")):
        if Kt % s:
            print(f"shards={s}: Kt {Kt} not divisible, skipped")
            continue
        shard_w = Kt // s
        core_set = ttnn.num_cores_to_corerangeset(s, grid, True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set, (M, shard_w * 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        try:
            tms, a_s = bench(lambda: ttnn.to_memory_config(a, mem), args.reps, device)
        except Exception as e:  # noqa: BLE001
            print(f"shards={s} (shard [{M}, {shard_w * 32}]): to_memory_config FAILED {str(e).splitlines()[0][:90]}")
            continue
        print(
            f"shards={s} (shard [{M}, {shard_w * 32}], {M * shard_w * 64 / 1024:.0f} KB per core): interleaved->sharded {tms * 1e3:.0f} us"
        )
        for w in sorted(
            {d for d in (1, 2, 3, 6, 9) if shard_w % d == 0 or d % shard_w == 0}
            & {d for d in range(1, 28) if Kt % d == 0},
            reverse=True,
        ):
            for obh in (int(v) for v in args.out_block_h.split(",")):
                try:
                    ms, out = bench(
                        lambda: ttnn.matmul(a_s, b, program_config=pc(w, obh), compute_kernel_config=cfg),
                        args.reps,
                        device,
                    )
                    err = float((ttnn.to_torch(out).float() - ref_h).abs().max())
                    ttnn.deallocate(out)
                    print(
                        f"  sharded in0 s={s}, w={w} {obh}x{per_core_n}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOP/s  max|diff| {err:.2e}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  sharded in0 s={s}, w={w} {obh}x{per_core_n}: FAILED {str(e).splitlines()[0][:110]}",
                        flush=True,
                    )
        ttnn.deallocate(a_s)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
