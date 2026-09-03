import argparse
import json
import time

import torch

import ttnn

TILE = 32
V, H = 154880, 2048


def _rect_grid(cores):
    for cols in range(min(cores, 11), 0, -1):
        if cores % cols == 0 and cores // cols <= 10:
            return cols, cores // cols
    raise ValueError(cores)


def lm_head_1d_pc(nt, kt, cores, bw):
    per_core_n = -(-nt // cores)
    blocks = -(-nt // per_core_n)
    cols, rows = _rect_grid(blocks)
    osw = max(d for d in (1, 2, 4) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=osw,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "dram"], required=True)
    ap.add_argument("--in0-block-w", type=int, default=4)
    ap.add_argument("--fidelity", choices=["lofi", "hifi2"], default="hifi2")
    ap.add_argument("--iters", type=int, default=32)
    args = ap.parse_args()

    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    try:
        banks = dev.dram_grid_size().x
        w = (torch.randn(H, V) * 0.02).contiguous()

        fidelity = ttnn.MathFidelity.LoFi if args.fidelity == "lofi" else ttnn.MathFidelity.HiFi2
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )

        x_torch = torch.randn(1, 1, 32, H) * 0.02

        if args.mode == "baseline":
            weight = ttnn.from_torch(
                w, device=dev, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            pc = lm_head_1d_pc(V // TILE, H // TILE, 110, args.in0_block_w)
            x = ttnn.from_torch(
                x_torch, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            out_mem = ttnn.L1_MEMORY_CONFIG
        else:
            assert V % (TILE * banks) == 0, (V, banks)
            n_pad = V
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            wspec = ttnn.ShardSpec(grid, (H, n_pad // banks), ttnn.ShardOrientation.ROW_MAJOR)
            wmem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, wspec)
            weight = ttnn.from_torch(w, device=dev, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=wmem)
            per_core_n = n_pad // TILE // banks
            kt = H // TILE
            bw = max(d for d in range(1, min(kt, args.in0_block_w) + 1) if kt % d == 0)
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=bw, per_core_M=1, per_core_N=per_core_n, fused_activation=None
            )
            # activation width-sharded across the same 8-core DRAM-bank raster
            in_mem = ttnn.create_sharded_memory_config(
                shape=(TILE, H // banks),
                core_grid=ttnn.num_cores_to_corerangeset(banks, dev.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.from_torch(x_torch, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            x = ttnn.to_memory_config(x, in_mem)
            out_mem = ttnn.create_sharded_memory_config(
                shape=(TILE, n_pad // banks),
                core_grid=ttnn.num_cores_to_corerangeset(banks, dev.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        out_c = ttnn.linear(x, weight, program_config=pc, memory_config=out_mem, compute_kernel_config=ck)
        ttnn.deallocate(out_c)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out_t = ttnn.linear(x, weight, program_config=pc, memory_config=out_mem, compute_kernel_config=ck)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t1 = time.perf_counter()
        us = (t1 - t0) / args.iters * 1e6
        assert not torch.isnan(ttnn.to_torch(out_t)).any()
        ttnn.release_trace(dev, tid)
        result = {
            "mode": args.mode,
            "in0_block_w": args.in0_block_w,
            "fidelity": args.fidelity,
            "us_per_call": round(us, 2),
        }
        print("RESULT", json.dumps(result))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
