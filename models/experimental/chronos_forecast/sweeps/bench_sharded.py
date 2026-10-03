# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""L1-interleaved (11x10) vs block-sharded (8x10) RMSNorm / add / QKV matmul for one 64-series chunk."""

import time

import torch
import ttnn

from models.experimental.chronos_forecast.tt import program_configs

M, D, N = 64 * 160, 768, 2304


def bench(fn, iters=5):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


if __name__ == "__main__":
    from tracy import signpost

    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    signpost("chronos_device_forward_start")
    try:
        x_h = torch.randn(1, 1, M, D)
        x = ttnn.from_torch(
            x_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        y = ttnn.from_torch(
            x_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        w = ttnn.from_torch(torch.ones(1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        wqkv = ttnn.from_torch(torch.randn(D, N) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        l1 = ttnn.L1_MEMORY_CONFIG

        print(
            f"interleaved rms_norm {bench(lambda: ttnn.rms_norm(x, weight=w, epsilon=1e-6, memory_config=l1)):8.1f} us"
        )
        print(f"interleaved add      {bench(lambda: ttnn.add(x, y, memory_config=l1)):8.1f} us")
        print(
            f"interleaved qkv      {bench(lambda: program_configs.linear(x, wqkv, memory_config=l1, dtype=ttnn.bfloat8_b)):8.1f} us"
        )

        gx, gy = 8, 10
        shard = ttnn.create_sharded_memory_config(
            (1, 1, M, D),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        xs = ttnn.to_memory_config(x, shard)
        ys = ttnn.to_memory_config(y, shard)
        ttnn.deallocate(x)
        ttnn.deallocate(y)
        block_h, block_w = M // 32 // gy, D // 32 // gx
        ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy), subblock_w=block_w, block_h=block_h, block_w=block_w, inplace=False
        )
        print(
            f"sharded rms_norm     {bench(lambda: ttnn.rms_norm(xs, weight=w, epsilon=1e-6, memory_config=shard, program_config=ln_cfg)):8.1f} us"
        )
        print(f"sharded add          {bench(lambda: ttnn.add(xs, ys, memory_config=shard)):8.1f} us")
        print(f"sharded->interleaved {bench(lambda: ttnn.sharded_to_interleaved(xs, l1)):8.1f} us")
        mm_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=block_w,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=16,
            out_block_w=N // 32 // gx,
            per_core_M=block_h,
            per_core_N=N // 32 // gx,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        qkv_shard = ttnn.create_sharded_memory_config(
            (1, 1, M, N),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        print(
            f"sharded qkv          {bench(lambda: ttnn.linear(xs, wqkv, program_config=mm_cfg, memory_config=qkv_shard, dtype=ttnn.bfloat8_b, compute_kernel_config=program_configs.compute_kernel_config())):8.1f} us"
        )
    finally:
        signpost("chronos_device_forward_stop")
        ttnn.close_device(dev)
