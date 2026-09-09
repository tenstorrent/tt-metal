"""Sweep program configs for the GDN decode recurrent matmul at the served shape.

Left is the per-(user, local head) query/key row [32, 12, M, 128]; right is that
pair's recurrent state [32, 12, 128, 128].  Batch dims collapse to 384.  One
device is enough: the shape is already TP-local.
"""
import argparse
import json
import time

import torch

import ttnn

BATCH, HEADS, KD, VD = 32, 12, 128, 128


def bench(device, left, right, program_config, ckc, iters=20):
    kwargs = dict(memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    if program_config is not None:
        kwargs["program_config"] = program_config
    out = ttnn.matmul(left, right, **kwargs)
    ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    started = time.perf_counter()
    for _ in range(iters):
        out = ttnn.matmul(left, right, **kwargs)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return 1e6 * (time.perf_counter() - started) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    ap.add_argument("--m", type=int, default=1, help="logical M rows in the left operand")
    args = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    results = []
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        ckc_lofi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        grid = device.compute_with_storage_grid_size()
        print(f"compute grid: {grid.x} x {grid.y}", flush=True)
        left_t = torch.randn(BATCH, HEADS, args.m, KD, dtype=torch.bfloat16)
        right_t = torch.randn(BATCH, HEADS, KD, VD, dtype=torch.bfloat16)
        left = ttnn.from_torch(
            left_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        right = ttnn.from_torch(
            right_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        m_tiles = max(1, args.m // 32)
        n_tiles = VD // 32
        k_tiles = KD // 32

        cases = [("auto (no program_config)", None, ckc)]
        cases.append(
            (
                "1D mcast grid 4x1 w4 (shipped)",
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(4, 1),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=True,
                ),
                ckc,
            )
        )
        # Batched reuse: the batch must divide the core count or the program
        # hangs (doc/prefill_general_optimizations), so only grids whose product
        # divides 384 are legal here.
        for gx, gy in [(8, 1), (8, 2), (4, 8), (6, 8), (8, 4), (8, 6), (8, 8), (11, 10)]:
            if gx > grid.x or gy > grid.y:
                continue
            if (BATCH * HEADS) % (gx * gy):
                print(f"skip reuse {gx}x{gy}: {BATCH * HEADS} % {gx * gy} != 0", flush=True)
                continue
            cases.append(
                (
                    f"reuse grid {gx}x{gy} ({gx * gy} cores)",
                    ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=k_tiles,
                        out_subblock_h=min(2, m_tiles),
                        out_subblock_w=min(4, n_tiles),
                        per_core_M=m_tiles,
                        per_core_N=n_tiles,
                    ),
                    ckc,
                )
            )
        cases.append(
            (
                "reuse grid 8x8 LoFi",
                ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=k_tiles,
                    out_subblock_h=min(2, m_tiles),
                    out_subblock_w=min(4, n_tiles),
                    per_core_M=m_tiles,
                    per_core_N=n_tiles,
                ),
                ckc_lofi,
            )
        )
        for name, pc, kernel in cases:
            try:
                us = bench(device, left, right, pc, kernel)
                results.append({"case": name, "us": us, "m": args.m})
                print(f"{name:44s} {us:9.1f} us", flush=True)
            except Exception as exc:
                print(f"{name:44s} FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                results.append({"case": name, "error": str(exc)[:300], "m": args.m})
    finally:
        ttnn.close_device(device)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
