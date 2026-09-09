"""What bandwidth can the decode projection matmul actually reach?

The shipped decode path runs every projection as a DRAM-sharded matmul over
``dram_grid_size().x`` cores (8 on p300c), M = 32 rows.  This measures that
against the alternatives at the two shapes that dominate the step: the MLP
gate/up (5120 -> 4352) and the packed GDN input (5120 -> 4160).
"""

import argparse
import json
import math
import time

import torch

import ttnn


def l1_width(*, rows, width, cores):
    return ttnn.create_sharded_memory_config(
        shape=(rows, math.ceil(width / cores / ttnn.TILE_SIZE) * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(x=cores, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def dram_weight_config(device, *, k, n):
    cores = device.dram_grid_size().x
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * cores)) * ttnn.TILE_SIZE * cores
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))}),
        (k, padded_n // cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)


def timed(fn, iters=30):
    fn()
    ttnn.synchronize_device(fn.device)
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(fn.device)
    return 1e6 * (time.perf_counter() - started) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    ap.add_argument("--rows", type=int, default=32)
    args = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    results = []
    try:
        grid = device.compute_with_storage_grid_size()
        dram_cores = device.dram_grid_size()
        print(f"compute grid {grid.x}x{grid.y}   dram grid {dram_cores.x}x{dram_cores.y}", flush=True)
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        rows = args.rows
        for k, n, label in [(5120, 4352, "mlp_gate/up"), (5120, 4160, "gdn_packed_in"), (4352, 5120, "mlp_down")]:
            weight_t = torch.randn(1, 1, k, n, dtype=torch.bfloat16)
            act_t = torch.randn(1, 1, rows, k, dtype=torch.bfloat16)
            act_dram = ttnn.from_torch(
                act_t,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # 1. Shipped: DRAM-sharded weights, 8 L1 width-sharded cores.
            cores = dram_cores.x
            w_shard = ttnn.from_torch(
                weight_t,
                device=device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_weight_config(device, k=k, n=n),
            )
            act_l1 = ttnn.to_memory_config(act_dram, l1_width(rows=rows, width=k, cores=cores))
            out_mc = l1_width(rows=rows, width=n, cores=cores)

            def shipped(w=w_shard, a=act_l1, mc=out_mc, k=k, n=n, cores=cores):
                out = ttnn.linear(
                    a,
                    w,
                    dtype=ttnn.bfloat16,
                    memory_config=mc,
                    compute_kernel_config=ckc,
                    program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=(k // ttnn.TILE_SIZE // cores),
                        per_core_M=1,
                        per_core_N=math.ceil(n / ttnn.TILE_SIZE / cores),
                    ),
                )
                ttnn.deallocate(out)

            shipped.device = device
            us = timed(shipped)
            bytes_read = k * n * 0.5625  # bfp4_b payload
            results.append(
                {
                    "shape": f"{rows}x{k}x{n}",
                    "label": label,
                    "case": f"dram-sharded {cores} cores (shipped)",
                    "us": us,
                    "gbps": bytes_read / us / 1000,
                }
            )
            print(
                f"{label:15s} {rows}x{k}x{n}  dram-sharded {cores}c  {us:8.1f} us  {bytes_read / us / 1000:7.1f} GB/s",
                flush=True,
            )

            # 2. Interleaved DRAM weights, 1D multicast over a big worker grid.
            w_inter = ttnn.from_torch(
                weight_t,
                device=device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for gx, gy in [(8, 8), (grid.x, grid.y), (8, 4), (11, 8)]:
                if gx > grid.x or gy > grid.y:
                    continue
                n_tiles = math.ceil(n / ttnn.TILE_SIZE)
                per_core_n = math.ceil(n_tiles / (gx * gy))
                k_tiles = k // ttnn.TILE_SIZE
                in0_block_w = 1
                for candidate in (8, 5, 4, 2, 1):
                    if k_tiles % candidate == 0:
                        in0_block_w = candidate
                        break
                sub_w = 4
                while sub_w > 1 and per_core_n % sub_w:
                    sub_w -= 1

                def mcast(
                    w=w_inter, a=act_dram, gx=gx, gy=gy, per_core_n=per_core_n, in0_block_w=in0_block_w, sub_w=sub_w
                ):
                    out = ttnn.linear(
                        a,
                        w,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ckc,
                        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=in0_block_w,
                            out_subblock_h=1,
                            out_subblock_w=sub_w,
                            per_core_M=1,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        ),
                    )
                    ttnn.deallocate(out)

                mcast.device = device
                try:
                    us = timed(mcast)
                    results.append(
                        {
                            "shape": f"{rows}x{k}x{n}",
                            "label": label,
                            "case": f"1D mcast {gx}x{gy} ({gx * gy} cores), interleaved",
                            "us": us,
                            "gbps": bytes_read / us / 1000,
                        }
                    )
                    print(
                        f"{label:15s} {rows}x{k}x{n}  1D mcast {gx}x{gy}  {us:8.1f} us  {bytes_read / us / 1000:7.1f} GB/s",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"{label:15s} 1D mcast {gx}x{gy} FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)

            # 3. Auto program selection, interleaved.
            def auto(w=w_inter, a=act_dram):
                out = ttnn.linear(
                    a, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc
                )
                ttnn.deallocate(out)

            auto.device = device
            us = timed(auto)
            results.append(
                {
                    "shape": f"{rows}x{k}x{n}",
                    "label": label,
                    "case": "auto, interleaved",
                    "us": us,
                    "gbps": bytes_read / us / 1000,
                }
            )
            print(
                f"{label:15s} {rows}x{k}x{n}  auto           {us:8.1f} us  {bytes_read / us / 1000:7.1f} GB/s",
                flush=True,
            )

            ttnn.deallocate(w_shard)
            ttnn.deallocate(w_inter)
            ttnn.deallocate(act_l1)
            ttnn.deallocate(act_dram)
    finally:
        ttnn.close_device(device)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
