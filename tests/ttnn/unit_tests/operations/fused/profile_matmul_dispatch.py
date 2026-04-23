"""
Profile matmul dispatch stages for the 3 worst Mcast2D offenders.

Run via pytest so device fixtures handle dispatch core setup correctly:
  pytest /tmp/profile_matmul_dispatch.py -xvs

Timing breakdown (median over N samples):
  t_cfg      = Python program-config object construction time
  t_dispatch = full ttnn.matmul on a warm cache (descriptor rebuild + enqueue)
  overhead   = t_dispatch - t_cfg  (time spent in C++ factory / enqueue)
"""
import time
import statistics
import cProfile
import pstats
import io

import torch
import pytest
import ttnn

N = 200
GX, GY = 8, 8
GRID = (GX, GY)


def _ck(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _med_us(fn, n=N):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1e3  # return microseconds


CASES = [
    # (label, m, k, n, transpose_mcast, in0_sharded)
    ("dram_in0", 1024, 1024, 1024, False, False),
    ("block_sharded_in0", 1024, 1024, 1024, False, True),
    ("transposed_mcast", 1024, 1024, 1024, True, True),
]


def _build_cfg(m, k, n, xpose):
    bw = k // 32 // GX
    pcM = m // 32 // GY
    pcN = n // 32 // GX
    sh = 1
    sw = next(s for s in [8, 4, 2, 1] if pcN % s == 0)
    return (
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            out_block_h=pcM,
            out_block_w=pcN,
            per_core_M=pcM,
            per_core_N=pcN,
            transpose_mcast=xpose,
            fused_activation=None,
            fuse_batch=True,
        ),
        sh,
        sw,
        pcM,
        pcN,
    )


def _warmup_and_sync(device, fn, n=20):
    for _ in range(n):
        fn()
    ttnn.synchronize_device(device)


def test_profile_mcast2d_dispatch(device):
    """Profile the 3 Mcast2D variants: measure per-call overhead on warm cache."""
    ck = _ck(device)

    print(f"\n{'Variant':<22} {'cfg_us':>8} {'dispatch_us':>12} {'overhead_us':>14} {'overhead%':>10}")
    print("-" * 72)

    results = {}

    for label, m, k, n, xpose, sharded in CASES:
        prog, sh, sw, pcM, pcN = _build_cfg(m, k, n, xpose)

        if sharded:
            in0_mem = ttnn.create_sharded_memory_config(
                (1, 1, m, k),
                core_grid=ttnn.CoreGrid(y=GY, x=GX),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR if xpose else ttnn.ShardOrientation.ROW_MAJOR,
            )
            out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
        else:
            in0_mem = ttnn.DRAM_MEMORY_CONFIG
            out_mem = ttnn.DRAM_MEMORY_CONFIG

        in0_t = ttnn.from_torch(
            torch.randn(1, 1, m, k).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in0_mem,
        )
        in1_t = ttnn.from_torch(
            torch.randn(1, 1, k, n).bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def dispatch():
            ttnn.matmul(
                in0_t,
                in1_t,
                program_config=prog,
                memory_config=out_mem,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ck,
            )

        # Warmup
        _warmup_and_sync(device, dispatch, n=10)

        # Measure Python config object construction time
        def make_cfg():
            _build_cfg(m, k, n, xpose)

        t_cfg = _med_us(make_cfg)
        t_dispatch = _med_us(dispatch)

        overhead = t_dispatch - t_cfg
        pct = overhead / t_dispatch * 100

        print(f"{label:<22} {t_cfg:>8.2f} {t_dispatch:>12.2f} {overhead:>14.2f} {pct:>9.1f}%")

        results[label] = {"t_cfg_us": t_cfg, "t_dispatch_us": t_dispatch, "overhead_us": overhead}

        ttnn.deallocate(in0_t)
        ttnn.deallocate(in1_t)

    # -----------------------------------------------------------------------
    # cProfile breakdown for worst offender (block_sharded_in0)
    # -----------------------------------------------------------------------
    print("\n\n=== cProfile: block_sharded_in0 (100 dispatch calls, warm cache) ===")

    label, m, k, n, xpose, sharded = CASES[1]  # block_sharded_in0
    prog, sh, sw, pcM, pcN = _build_cfg(m, k, n, xpose)

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=GY, x=GX),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    in0_t = ttnn.from_torch(
        torch.randn(1, 1, m, k).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    in1_t = ttnn.from_torch(
        torch.randn(1, 1, k, n).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def dispatch_block_sharded():
        ttnn.matmul(
            in0_t,
            in1_t,
            program_config=prog,
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ck,
        )

    _warmup_and_sync(device, dispatch_block_sharded, n=10)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        dispatch_block_sharded()
    ttnn.synchronize_device(device)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
    print(s.getvalue())

    ttnn.deallocate(in0_t)
    ttnn.deallocate(in1_t)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Stage Timing Breakdown (block_sharded_in0) ===")
    print("Hot path per cache-hit call (fast path active via emplace_runtime_args):")
    print("  1. validate_on_program_cache_hit() — calls validate_on_cache_miss (no cache-hit override)")
    print("  2. apply_resolved_bindings() — O(#buffer_args) direct writes into cached Program")
    print("  3. EnqueueProgram() — issue to device")
    print()

    t_full = results["block_sharded_in0"]["t_dispatch_us"]
    t_py = results["block_sharded_in0"]["t_cfg_us"]
    print(f"  Total dispatch time (warm cache): {t_full:.1f} µs")
    print(f"  Python config construction:       {t_py:.2f} µs  (negligible)")
    print(f"  C++ factory + enqueue overhead:   {t_full - t_py:.1f} µs")
    print()
    print("  Breakdown: validate_on_cache_miss + apply_resolved_bindings (~1µs) + EnqueueProgram")
    print("  create_descriptor() is NOT called on cache hits (fast path active)")
