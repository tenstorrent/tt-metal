import time

import torch

import ttnn

TILE = 32
H, NE = 2048, 64


def rect_grid(cores):
    for cols in range(min(cores, 11), 0, -1):
        if cores % cols == 0 and cores // cols <= 10:
            return cols, cores // cols
    raise ValueError(cores)


def mcast_1d_pc(nt, kt, target_cores, osw_cap=1):
    per_core_n = -(-nt // target_cores)
    blocks = -(-nt // per_core_n)
    cols, rows = rect_grid(blocks)
    in0_bw = max(d for d in range(1, min(kt, 48) + 1) if kt % d == 0)
    osw = max(d for d in range(1, osw_cap + 1) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        in0_block_w=in0_bw,
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


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--fidelity", choices=["hifi4", "hifi2"], default="hifi4")
args = ap.parse_args()

dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
try:
    w = (torch.randn(H, NE) * 0.02).contiguous()
    weight = ttnn.from_torch(
        w, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pc = mcast_1d_pc(NE // TILE, H // TILE, 2, osw_cap=1)
    fidelity = ttnn.MathFidelity.HiFi4 if args.fidelity == "hifi4" else ttnn.MathFidelity.HiFi2
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    x = torch.randn(1, 1, 32, H) * 0.02
    x_dev = ttnn.from_torch(
        x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    out_c = ttnn.linear(x_dev, weight, program_config=pc, compute_kernel_config=ck, dtype=ttnn.float32)
    ttnn.deallocate(out_c)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out_t = ttnn.linear(x_dev, weight, program_config=pc, compute_kernel_config=ck, dtype=ttnn.float32)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(32):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t1 = time.perf_counter()
    us = (t1 - t0) / 32 * 1e6
    ttnn.release_trace(dev, tid)
    print(f"RESULT fidelity={args.fidelity} us_per_call={us:.2f}")
finally:
    ttnn.close_device(dev)
