# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm over [M, 2048] at small M: stock interleaved vs block-sharded (with the layout conversions).

Stock rms_norm on an interleaved tensor gives each core whole tile-rows, so M = 512
(16 tile-rows) runs on 16 cores. The sharded variant splits the width as well.
"""

import argparse
import time

import torch

import ttnn


def timed(device, fn, reps=20):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    t = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t) / reps * 1e6
    ttnn.release_trace(device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[128, 512, 2048])
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    W = 2048
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    try:
        torch.manual_seed(0)
        g = torch.rand(W) + 0.5
        gw = ttnn.from_torch(
            g.reshape(1, 1, W // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        for M in args.m:
            xt = torch.randn(1, 1, M, W)
            ref = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-5) * g
            x = ttnn.from_torch(
                xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            stock = lambda: ttnn.rms_norm(
                x, epsilon=1e-5, weight=gw, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            err = (ttnn.to_torch(stock()).float() - ref).abs().max().item()
            print(f"M={M:<5} stock interleaved      {timed(dev, stock):7.1f} us  err {err:.3g}", flush=True)
            mt, wt = M // 32, W // 32
            for gx, gy in ((8, 8), (8, 4), (16 // 2, 2), (4, 4), (8, 1), (13, 10)):
                if (
                    mt % gy
                    or wt % gx
                    or gx > dev.compute_with_storage_grid_size().x
                    or gy > dev.compute_with_storage_grid_size().y
                ):
                    continue
                bh, bw = mt // gy, wt // gx
                shard = ttnn.create_sharded_memory_config(
                    shape=(M // gy, W // gx),
                    core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    subblock_w=min(bw, 4) if bw % min(bw, 4) == 0 else 1,
                    block_h=bh,
                    block_w=bw,
                    inplace=False,
                )

                def sharded():
                    xs = ttnn.interleaved_to_sharded(x, shard)
                    y = ttnn.rms_norm(
                        xs, epsilon=1e-5, weight=gw, program_config=pc, compute_kernel_config=ckc, memory_config=shard
                    )
                    ttnn.deallocate(xs)
                    out = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(y)
                    return out

                try:
                    err = (ttnn.to_torch(sharded()).float() - ref).abs().max().item()
                    print(
                        f"M={M:<5} block-sharded {gx}x{gy:<2}     {timed(dev, sharded):7.1f} us  err {err:.3g}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"M={M:<5} block-sharded {gx}x{gy:<2}     ERR {str(e)[:100]}", flush=True)
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
