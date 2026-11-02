# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: K-block depth sweep of MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0) at the TP shard
shapes, HiFi4 / fp32 accumulate. The iteration-34 script's L1 estimate counted the double buffering twice
(4 x tile bytes), which hid every in0_block_w above 3 on the 11x10 grid; this copy uses the factory's allocation
(in0 / in1 blocks double-buffered in bf16, the fp32 partial block, the bf16 output block) and adds
``--no-l1-acc`` (partials reloaded through DEST instead of the packer's L1 accumulate) and ``--out-blocks``.

    python iter35_matmul_sweep1d.py --m 1024 --k 5184 --n 16384 --grids 11x10 --in0-block-w 6,9,18,27
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import ttnn


FULL_SYNC = False
L1_ACC = True


def cfg():
    # dst_full_sync_en (iteration 34): the fp32 DEST holds 8 tiles per acquire instead of 4, so output subblocks
    # of up to 8 tiles are legal; the accumulation itself is unchanged.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        packer_l1_acc=L1_ACC,
        dst_full_sync_en=FULL_SYNC,
    )


def dev(a, device):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / reps, out


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--k", type=int, default=5184)
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--transpose-b", action="store_true")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--full-sync", action="store_true", help="dst_full_sync_en: 8 fp32 DEST tiles, subblocks up to 8")
    p.add_argument("--top", type=int, default=20)
    p.add_argument(
        "--in0-block-w", default=None, help="comma list to restrict in0_block_w (default: every divisor of Kt up to 27)"
    )
    p.add_argument("--subs", default=None, help="comma list of out_subblock HxW to restrict, e.g. 4x1,2x2,1x4")
    p.add_argument(
        "--ref-grid", default="11x8", help="core grid XxY of the reference (what the model uses for the shape)"
    )
    p.add_argument("--grids", default=None, help="comma list of grids XxY to restrict, e.g. 10x10,11x9")
    p.add_argument("--out-blocks", default=None, help="comma list of out_block HxW to restrict, e.g. 16x5,32x5")
    p.add_argument("--no-l1-acc", action="store_true", help="packer_l1_acc=False: partials reloaded through DEST")
    p.add_argument("--l1-limit", type=int, default=1_400_000, help="L1 bytes allowed for the CB estimate")
    args = p.parse_args()
    global FULL_SYNC, L1_ACC
    FULL_SYNC = args.full_sync
    L1_ACC = not args.no_l1_acc
    only_ob = {tuple(int(v) for v in ob.split("x")) for ob in args.out_blocks.split(",")} if args.out_blocks else None
    only_w = {int(v) for v in args.in0_block_w.split(",")} if args.in0_block_w else None
    only_subs = {tuple(int(v) for v in sw.split("x")) for sw in args.subs.split(",")} if args.subs else None
    only_grids = {tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")} if args.grids else None
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    rng = np.random.default_rng(0)
    a = dev(rng.standard_normal((1, 1, M, K)).astype(np.float32), device)
    b = dev((rng.standard_normal((1, 1, N, K) if args.transpose_b else (1, 1, K, N)) * 0.02).astype(np.float32), device)
    flops = 2 * M * K * N
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    rx, ry = (int(v) for v in args.ref_grid.split("x"))
    ms, ref = bench(
        lambda: ttnn.matmul(
            a, b, transpose_b=args.transpose_b, compute_kernel_config=cfg(), core_grid=ttnn.CoreGrid(y=ry, x=rx)
        ),
        args.reps,
        device,
    )
    ref_host = ttnn.to_torch(ref).float()
    print(
        f"M={M} K={K} N={N}: ttnn's own choice on the {rx}x{ry} grid {ms:.2f} ms, {flops / ms / 1e9:.1f} TFLOP/s wall"
    )
    results, tried = [], 0
    for gx, gy in ((11, 8), (11, 10), (10, 10), (11, 9), (10, 8), (8, 8)):
        if only_grids is not None and (gx, gy) not in only_grids:
            continue
        cores = gx * gy
        per_core_n = -(-Nt // cores)
        per_core_m = Mt
        for in0_block_w in [d for d in divisors(Kt) if d <= 27 and (only_w is None or d in only_w)]:
            for sub_h, sub_w in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4), (4, 2), (2, 4), (8, 1), (1, 8)):
                if sub_h * sub_w > (8 if FULL_SYNC else 4):  # fp32 DEST tiles per acquire (8 in full-sync mode)
                    continue
                if only_subs is not None and (sub_h, sub_w) not in only_subs:
                    continue
                for out_block_h in [d for d in divisors(per_core_m) if d % sub_h == 0 and d >= sub_h]:
                    for out_block_w in [d for d in divisors(per_core_n) if d % sub_w == 0 and d >= sub_w]:
                        if only_ob is not None and (out_block_h, out_block_w) not in only_ob:
                            continue
                        # L1 estimate as the factory allocates: in0 block (out_block_h x in0_block_w) and in1 block
                        # (in0_block_w x out_block_w) double-buffered in bf16 (2 KB tiles), the fp32 partial block
                        # (4 KB tiles) and the bf16 output block; skip what cannot fit.
                        est = (
                            2 * 2048 * (out_block_h * in0_block_w + in0_block_w * out_block_w)
                            + (4096 + 2048) * out_block_h * out_block_w
                        )
                        if est > args.l1_limit:
                            continue
                        tried += 1
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=in0_block_w,
                            out_subblock_h=sub_h,
                            out_subblock_w=sub_w,
                            out_block_h=out_block_h,
                            out_block_w=out_block_w,
                            per_core_M=per_core_m,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        )
                        name = f"grid {gx}x{gy} per_core_N={per_core_n} in0_block_w={in0_block_w} sub={sub_h}x{sub_w} out_block={out_block_h}x{out_block_w} l1_acc={int(L1_ACC)}"
                        try:
                            t, out = bench(
                                lambda: ttnn.matmul(
                                    a, b, transpose_b=args.transpose_b, program_config=pc, compute_kernel_config=cfg()
                                ),
                                args.reps,
                                device,
                            )
                            err = float((ttnn.to_torch(out).float() - ref_host).abs().max())
                            ttnn.deallocate(out)
                            results.append((t, name, err))
                            print(
                                f"  {t:8.2f} ms  {flops / t / 1e9:7.1f} TFLOP/s  max|diff| {err:.2e}  {name}",
                                flush=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            results.append((1e9, name + " FAILED " + str(e).splitlines()[0][:70], 0.0))
    print("--- sorted ---", flush=True)
    results.sort(key=lambda r: r[0])
    ok = [r for r in results if r[0] < 1e9]
    print(f"{tried} configs tried, {len(ok)} ran, {len(results) - len(ok)} failed")
    for t, name, err in ok[: args.top]:
        print(f"  {t:8.2f} ms  {flops / t / 1e9:7.1f} TFLOP/s  max|diff| {err:.2e}  {name}")
    for t, name, err in [r for r in results if r[0] >= 1e9][:3]:
        print("  ", name[:160])
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
