#!/usr/bin/env python3
"""
Standalone profiling script for unified matmul helper.
Runs matmul through the non-multicast ReuseProgramConfig factory
(which uses ROW_MAJOR_OUTPUT / matmul_blocks_absolute).

Usage:
    unset TT_METAL_DPRINT_CORES && python -m tracy -r -v profile_matmul_unified.py

Then extract results:
    csvfile=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    awk -F',' 'NR==1 || /matmul/ {printf "%-20s | DevKernel: %10s ns\n", $1, $19}' "$csvfile"
"""
import torch
import ttnn


def find_max_subblock(out_block_h, out_block_w, max_dst=8):
    best_h, best_w, best_area = 1, 1, 1
    for sh in range(1, out_block_h + 1):
        if out_block_h % sh != 0:
            continue
        for sw in range(1, out_block_w + 1):
            if out_block_w % sw != 0:
                continue
            if sh * sw <= max_dst and sh * sw >= best_area:
                best_h, best_w, best_area = sh, sw, sh * sw
    return best_h, best_w, best_area


def run_matmul(device, m, k, n, num_warmup=2, num_iters=5):
    """Run matmul via MatmulMultiCoreReuseProgramConfig (non-multicast, unified helper)."""
    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = torch.randn(1, 1, k, n).bfloat16()

    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_block_h = m // 32
    out_block_w = n // 32
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=k // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
    )

    # Warmup (fills program cache)
    for _ in range(num_warmup):
        out_t = ttnn.matmul(in0_t, in1_t, program_config=program_config)
        ttnn.deallocate(out_t)

    # Measured iterations
    for _ in range(num_iters):
        out_t = ttnn.matmul(in0_t, in1_t, program_config=program_config)
        ttnn.deallocate(out_t)

    ttnn.deallocate(in0_t)
    ttnn.deallocate(in1_t)
    print(f"  {m}x{k}x{n}: {num_iters} iterations complete")


def main():
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    shapes = [
        (256, 512, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 512, 2048),
        (2048, 1024, 2048),
    ]

    print("Profiling unified matmul helper (MatmulMultiCoreReuse factory):")
    for m, k, n in shapes:
        run_matmul(device, m, k, n)

    ttnn.close_device(device)
    print("Done. Check generated/profiler/reports/ for Tracy output.")


if __name__ == "__main__":
    main()
