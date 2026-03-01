# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion Infrastructure Demo Suite

Demos showcasing different fusion capabilities:

1. Basic 3-op chain (RMS -> Matmul -> RMS) with DRAM I/O on a 4x2 grid
2. Sharded 2-op chain (RMS -> LN) demonstrating pinned buffer address reassignment
4. Two parallel sharded chains (LN→MM, RMS→MM) on disjoint 1×8 core columns
5. GlobalCircularBuffer mid-kernel write to an external consumer
8. Sharded heterogeneous tree (LN → Slice → Matmul → Slice → LN) with block-sharded L1 intermediates

Each demo is split into separate fused and unfused tests, each with cold + warm timing.
Cold = all caches cleared (JIT disk + in-memory + program + fusion build),
warm = all caches populated.
"""

import shutil
import time
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion import clear_build_cache


# =============================================================================
# Helpers
# =============================================================================


COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
)


def _clear_all_caches(device):
    """Clear every cache layer for a truly cold dispatch.

    Clears: fusion build cache, device program cache, in-memory JIT cache,
    and disk-cached compiled kernels. Preserves firmware/ ELFs (needed by the
    running process's linker for --just-symbols).
    """
    clear_build_cache()  # fusion Python build cache
    device.clear_program_cache()  # device program cache
    ttnn.device.ClearKernelCache()  # in-memory JIT build cache
    cache_dir = Path.home() / ".cache" / "tt-metal-cache"
    for kernels_dir in cache_dir.glob("*/*/kernels"):
        shutil.rmtree(kernels_dir)


def _time_cold_warm(cold_fn, device, warm_fn=None):
    """Clear all caches, run cold_fn() once, then run warm_fn() once.

    If warm_fn is None, cold_fn is used for both.
    Returns (cold_ms, warm_ms).
    """
    if warm_fn is None:
        warm_fn = cold_fn
    sync = lambda: ttnn.synchronize_device(device)
    _clear_all_caches(device)
    sync()
    t0 = time.perf_counter()
    cold_fn()
    sync()
    cold = 1000 * (time.perf_counter() - t0)

    sync()
    t0 = time.perf_counter()
    warm_fn()
    sync()
    warm = 1000 * (time.perf_counter() - t0)
    return cold, warm


def _time_fused(build_fn, device):
    """Build + time cold/warm for a fused op.

    build_fn: callable returning a FusedOp (e.g. ``lambda: Sequential(...).build(device)``)
    Returns (fused_op, cold_ms, warm_ms).
    """
    fused = [None]

    def build_and_launch():
        fused[0] = build_fn()
        fused[0].launch()

    cold, warm = _time_cold_warm(build_and_launch, device, warm_fn=lambda: fused[0].launch())
    return fused[0], cold, warm


def _time_steady_state(fn, device, num_warmup=5, num_measure=100):
    """Measure steady-state e2e time per iteration.

    Runs num_warmup iterations (discarded), then num_measure iterations
    timed as a batch, returning total_ms / num_measure.
    All caches are warm (program cache, build cache, JIT cache).
    """
    sync = lambda: ttnn.synchronize_device(device)

    # Warmup
    for _ in range(num_warmup):
        fn()
    sync()

    # Measure
    t0 = time.perf_counter()
    for _ in range(num_measure):
        fn()
    sync()
    total_ms = 1000 * (time.perf_counter() - t0)
    return total_ms / num_measure


# =============================================================================
# Tests
# =============================================================================


class TestFusedDemo:
    """Fusion infrastructure demo tests.

    Each demo is split into fused and unfused tests with cold + warm timing.
    Cold = all caches cleared, warm = all caches populated.
    """

    # Set to True for Tracy profiling: skips timing loops, runs once only.
    _SINGLE_RUN_ONLY = False

    # -----------------------------------------------------------------
    # Demo 1: RMS -> Matmul -> RMS (DRAM, 4x2 grid)
    # Input [256, H], matmul [H, H], 4x2 = 8 cores
    # H must be divisible by 128 (in0_block_w=4 × tile=32)
    # -----------------------------------------------------------------

    def _demo1_setup(self, device, H):
        torch.manual_seed(42)
        M_tiles = 256 // 32  # 8
        N_tiles = H // 32
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=min(N_tiles, 4),
            per_core_M=M_tiles // 8,
            per_core_N=N_tiles,
        )

        torch_input = torch.randn(1, 1, 256, H, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, H, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, H, H, dtype=torch.bfloat16)

        return core_range, mm_cfg, torch_input, torch_w, torch_b

    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_demo1_fused(self, device, H):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        core_range, mm_cfg, torch_input, torch_w, torch_b = self._demo1_setup(device, H)

        dram = ttnn.DRAM_MEMORY_CONFIG
        tt_in = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        tt_w = ttnn.from_torch(torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
        # fp32 doesn't fit in L1 for H=1536 on the small 4x2 DRAM-interleaved grid
        compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )
        r1 = rms_norm.rms_norm(
            tt_in,
            core_range_set=core_range,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
        )
        m = matmul_desc(
            r1.output_tensors[0],
            tt_b,
            core_range_set=core_range,
            program_config=mm_cfg,
            compute_kernel_config=compute_cfg,
        )
        r2 = rms_norm.rms_norm(
            m.output_tensors[0],
            core_range_set=core_range,
            weight=ttnn.from_torch(
                torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
            ),
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
        )

        if self._SINGLE_RUN_ONLY:
            fused = Sequential(r1, m, r2).build(device)
            fused.launch()
            ttnn.synchronize_device(device)
            print(f"\n  Demo 1 Fused (H={H}): single run (for Tracy)")
        else:
            _, cold, warm = _time_fused(lambda: Sequential(r1, m, r2).build(device), device)

            fused = Sequential(r1, m, r2).build(device)
            e2e = _time_steady_state(fused.launch, device)

            fused_result = ttnn.to_torch(r2.output_tensors[0])

            # Unfused reference for PCC
            tt_in = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
            )
            tt_w = ttnn.from_torch(
                torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
            )
            tt_B = ttnn.from_torch(
                torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
            )
            u1 = ttnn.rms_norm(tt_in, weight=tt_w, epsilon=1e-5)
            u2 = ttnn.matmul(u1, tt_B, program_config=mm_cfg)
            ref = ttnn.to_torch(ttnn.rms_norm(u2, weight=tt_w, epsilon=1e-5))

            passing, pcc = comp_pcc(ref, fused_result, pcc=0.97)
            print(f"\n  Demo 1 Fused (H={H}): cold={cold:.2f}ms  e2e={e2e:.3f}ms  PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_demo1_unfused(self, device, H):
        core_range, mm_cfg, torch_input, torch_w, torch_b = self._demo1_setup(device, H)
        dram = ttnn.DRAM_MEMORY_CONFIG
        tt_in = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        tt_w = ttnn.from_torch(torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
        tt_B = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

        def unfused():
            u1 = ttnn.rms_norm(tt_in, weight=tt_w, epsilon=1e-5)
            u2 = ttnn.matmul(u1, tt_B, program_config=mm_cfg)
            return ttnn.rms_norm(u2, weight=tt_w, epsilon=1e-5)

        if self._SINGLE_RUN_ONLY:
            unfused()
            ttnn.synchronize_device(device)
            print(f"\n  Demo 1 Unfused (H={H}): single run (for Tracy)")
        else:
            cold, warm = _time_cold_warm(unfused, device)
            e2e = _time_steady_state(unfused, device)
            print(f"\n  Demo 1 Unfused (H={H}): cold={cold:.2f}ms  e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Demo 2: RMS -> LN (block-sharded, 4x4 grid)
    # Input [H, 512] on 4x4=16 cores, shard [H/4, 128]
    # H must be divisible by 128 (4 grid-rows × tile=32)
    # -----------------------------------------------------------------

    def _demo2_setup(self, device, H):
        torch.manual_seed(42)
        cols = 512
        shard_h = H // 4  # rows distributed across 4 core-rows
        shard_w = cols // 4  # 128 always
        block_h = shard_h // 32  # tile rows per core
        block_w = shard_w // 32  # tile cols per core = 4 always

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
        shard_spec = ttnn.ShardSpec(cores, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=4,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )

        torch_input = torch.randn(1, 1, H, cols, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, cols, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem,
        )
        # Norm weight [1,1,1,512] width-sharded across 4 columns → [32,128] per core
        w_shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        tt_w = ttnn.from_torch(
            torch_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard),
        )

        return cores, sharded_mem, program_cfg, tt_input, tt_w

    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_demo2_fused(self, device, H):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm

        cores, sharded_mem, program_cfg, tt_input, tt_w = self._demo2_setup(device, H)

        r = rms_norm.rms_norm(
            tt_input,
            core_range_set=cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=sharded_mem,
        )
        ln = layer_norm.layer_norm(
            r.output_tensors[0],
            core_range_set=cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=sharded_mem,
        )

        if self._SINGLE_RUN_ONLY:
            fused = Sequential(r, ln).build(device)
            fused.launch()
            ttnn.synchronize_device(device)
            print(f"\n  Demo 2 Fused (H={H}): single run (for Tracy)")
        else:
            _, cold, warm = _time_fused(lambda: Sequential(r, ln).build(device), device)

            fused = Sequential(r, ln).build(device)
            e2e = _time_steady_state(fused.launch, device)

            fused_result = ttnn.to_torch(ln.output_tensors[0])

            # Unfused reference for PCC
            u1 = ttnn.rms_norm(
                tt_input,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=sharded_mem,
            )
            ref = ttnn.to_torch(
                ttnn.layer_norm(
                    u1,
                    weight=tt_w,
                    epsilon=1e-5,
                    program_config=program_cfg,
                    compute_kernel_config=COMPUTE_CONFIG,
                    memory_config=sharded_mem,
                )
            )

            passing, pcc = comp_pcc(ref, fused_result, pcc=0.98)
            print(f"\n  Demo 2 Fused (H={H}): cold={cold:.2f}ms  e2e={e2e:.3f}ms  PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_demo2_unfused(self, device, H):
        cores, sharded_mem, program_cfg, tt_input, tt_w = self._demo2_setup(device, H)

        def unfused():
            u1 = ttnn.rms_norm(
                tt_input,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=sharded_mem,
            )
            return ttnn.layer_norm(
                u1,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=sharded_mem,
            )

        if self._SINGLE_RUN_ONLY:
            unfused()
            ttnn.synchronize_device(device)
            print(f"\n  Demo 2 Unfused (H={H}): single run (for Tracy)")
        else:
            cold, warm = _time_cold_warm(unfused, device)
            e2e = _time_steady_state(unfused, device)
            print(f"\n  Demo 2 Unfused (H={H}): cold={cold:.2f}ms  e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Demo 4: Two parallel sharded chains on disjoint 1×8 core columns
    # Chain A: LN -> Matmul on cores col 0, rows 0-7
    # Chain B: RMS -> Matmul on cores col 1, rows 0-7
    # Block-sharded [1024,256] inputs, sharded [1024,128] outputs on
    # 1×8 grid. B weight + norm weight/bias L1 sharded.
    # -----------------------------------------------------------------

    def _demo4_setup(self, device):
        torch.manual_seed(42)
        rows, K, N = 1024, 256, 128

        # 1-column grids: K (width) not split → matmul factory CB fits in shard
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))})

        shard_h = rows // 8  # 128

        def _shard_mem(cores, width):
            spec = ttnn.ShardSpec(cores, [shard_h, width], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        sharded_in_a = _shard_mem(cores_a, K)  # input shard [128, 256]
        sharded_in_b = _shard_mem(cores_b, K)  # input shard [128, 256]
        sharded_out_a = _shard_mem(cores_a, N)  # output shard [128, 128]
        sharded_out_b = _shard_mem(cores_b, N)  # output shard [128, 128]

        # [1024,256] × [256,128] on 1×8 grid
        # M=32 tiles, K=8 tiles, N=4 tiles
        # per_core_M=32/8=4, per_core_N=4, in0_block_w=K/32=8
        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 8),
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=min(N // 32, 4),
            per_core_M=shard_h // 32,
            per_core_N=N // 32,
        )

        torch_a = torch.randn(1, 1, rows, K, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, rows, K, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, K, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, K, dtype=torch.bfloat16)
        torch_B = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

        # Inputs block-sharded in L1
        ta = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_in_a
        )
        tb = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_in_b
        )

        # Norm weight/bias [1,1,1,256] width-sharded across 8 cores → [32,32] per core
        w_shard = ttnn.ShardSpec(cores_a, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
        tw = ttnn.from_torch(torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem)
        tbi = ttnn.from_torch(
            torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem
        )

        # Matmul B [1,1,256,128] must be interleaved for MatmulMultiCoreReuseProgramConfig
        tB = ttnn.from_torch(
            torch_B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return (
            cores_a,
            cores_b,
            sharded_in_a,
            sharded_in_b,
            sharded_out_a,
            sharded_out_b,
            mm_cfg,
            ta,
            tb,
            tw,
            tbi,
            tB,
        )

    def test_demo4_fused(self, device):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        (
            cores_a,
            cores_b,
            sharded_in_a,
            sharded_in_b,
            sharded_out_a,
            sharded_out_b,
            mm_cfg,
            ta,
            tb,
            tw,
            tbi,
            tB,
        ) = self._demo4_setup(device)

        la = layer_norm.layer_norm(
            ta,
            core_range_set=cores_a,
            weight=tw,
            bias=tbi,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=sharded_in_a,
        )
        ma = matmul_desc(
            la.output_tensors[0],
            tB,
            core_range_set=cores_a,
            program_config=mm_cfg,
            compute_kernel_config=COMPUTE_CONFIG,
            output_mem_config=sharded_out_a,
        )
        rb = rms_norm.rms_norm(
            tb,
            core_range_set=cores_b,
            weight=tw,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=sharded_in_b,
        )
        mb = matmul_desc(
            rb.output_tensors[0],
            tB,
            core_range_set=cores_b,
            program_config=mm_cfg,
            compute_kernel_config=COMPUTE_CONFIG,
            output_mem_config=sharded_out_b,
        )

        if self._SINGLE_RUN_ONLY:
            fused = Parallel(Sequential(la, ma), Sequential(rb, mb)).build(device)
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Demo 4 Fused: single run (for Tracy)")
        else:
            _, cold, warm = _time_fused(lambda: Parallel(Sequential(la, ma), Sequential(rb, mb)).build(device), device)

            fused = Parallel(Sequential(la, ma), Sequential(rb, mb)).build(device)
            e2e = _time_steady_state(fused.launch, device)

            result_a = ttnn.to_torch(ma.output_tensors[0])
            result_b = ttnn.to_torch(mb.output_tensors[0])

            # Unfused reference for PCC — interleaved to avoid core mapping constraints
            ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ua2 = ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)
            ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ub2 = ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)

            p_a, pcc_a = comp_pcc(ttnn.to_torch(ua2), result_a, pcc=0.97)
            p_b, pcc_b = comp_pcc(ttnn.to_torch(ub2), result_b, pcc=0.97)

            print(f"\n  Demo 4 Fused: cold={cold:.2f}ms  e2e={e2e:.3f}ms  PCC: a={pcc_a:.4f} b={pcc_b:.4f}")
            assert p_a, f"Chain A PCC: {pcc_a}"
            assert p_b, f"Chain B PCC: {pcc_b}"

    def test_demo4_unfused(self, device):
        """Unfused path using ttnn ops with sharded intermediates.

        Both chains serialize on the same (0,0)-based 1×8 grid with matching
        shard shapes (the fused path runs them in parallel on disjoint grids).
        """
        torch.manual_seed(42)
        rows, K, N = 1024, 256, 128
        shard_h = rows // 8  # 128

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})

        def _shard_mem(width):
            spec = ttnn.ShardSpec(cores, [shard_h, width], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        sharded_in = _shard_mem(K)  # [128, 256]
        sharded_out = _shard_mem(N)  # [128, 128]

        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 8),
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=min(N // 32, 4),
            per_core_M=shard_h // 32,
            per_core_N=N // 32,
        )
        ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(1, 8),
            subblock_w=min(K // 32, 4),
            block_h=shard_h // 32,
            block_w=K // 32,
            inplace=False,
        )

        ta = ttnn.from_torch(
            torch.randn(1, 1, rows, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_in,
        )
        tb = ttnn.from_torch(
            torch.randn(1, 1, rows, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_in,
        )
        # Norm weight/bias [1,1,1,256] width-sharded across 8 cores → [32,32] per core
        w_shard = ttnn.ShardSpec(cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
        tw = ttnn.from_torch(
            torch.ones(1, 1, 1, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=w_mem,
        )
        tbi = ttnn.from_torch(
            torch.zeros(1, 1, 1, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=w_mem,
        )
        # Matmul B [1,1,256,128] must be interleaved for MatmulMultiCoreReuseProgramConfig
        tB = ttnn.from_torch(
            torch.randn(1, 1, K, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def unfused():
            ua1 = ttnn.layer_norm(
                ta,
                weight=tw,
                bias=tbi,
                epsilon=1e-5,
                program_config=ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=sharded_in,
            )
            ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG, memory_config=sharded_out)
            ub1 = ttnn.rms_norm(
                tb,
                weight=tw,
                epsilon=1e-5,
                program_config=ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=sharded_in,
            )
            ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG, memory_config=sharded_out)

        if self._SINGLE_RUN_ONLY:
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Demo 4 Unfused: single run (for Tracy)")
        else:
            cold, warm = _time_cold_warm(unfused, device)
            e2e = _time_steady_state(unfused, device)
            print(f"\n  Demo 4 Unfused: cold={cold:.2f}ms  e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Demo 5: GlobalCircularBuffer mid-kernel write (fused only)
    # -----------------------------------------------------------------

    def test_demo5_fused(self, device):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        TILE_SIZE_BF16 = 2048  # 32x32 x 2 bytes

        DRAM_READER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_in");
    uint32_t tile_bytes = get_tile_size(cb_id);
    DataFormat data_format = get_dataformat(cb_id);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
"""

        TILE_COPY_COMPUTE_SOURCE = """\
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        cb_pop_front(cb_in, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
"""

        GLOBALCB_SENDER_WRITER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "api/remote_circular_buffer.h"
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t local_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t remote_cb_id = get_named_compile_time_arg_val("cb_remote");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    experimental::CircularBuffer local_cb{local_cb_id};
    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};
    experimental::Noc noc;
    remote_cb.set_receiver_page_size(noc, page_size);
    for (uint32_t i = 0; i < num_tiles; i++) {
        local_cb.wait_front(1);
        remote_cb.reserve_back(1);
        remote_cb.push_back(noc, local_cb, 1, 1, 1, page_size);
        local_cb.pop_front(1);
    }
    remote_cb.commit();
}
"""

        GLOBALCB_RECEIVER_READER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "api/remote_circular_buffer.h"
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t remote_cb_id = get_named_compile_time_arg_val("cb_remote");
    constexpr uint32_t local_cb_id = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    experimental::CircularBuffer local_cb{local_cb_id};
    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};
    experimental::Noc noc;
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    remote_cb.set_sender_page_size(noc, page_size);
    experimental::align_local_cbs_to_remote_cb<1>(remote_cb_id, {local_cb_id});
    for (uint32_t i = 0; i < num_tiles; i++) {
        local_cb.reserve_back(1);
        remote_cb.wait_front(1);
        local_cb.push_back(1);
        remote_cb.pop_front(noc, 1);
    }
    remote_cb.commit();
}
"""

        DRAM_WRITER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_out");
    uint32_t tile_bytes = get_tile_size(cb_id);
    DataFormat data_format = get_dataformat(cb_id);
    const InterleavedAddrGenFast<true> d = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write_tile(i, d, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
"""

        RECEIVER_DRAM_WRITER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_in");
    uint32_t tile_bytes = get_tile_size(cb_id);
    DataFormat data_format = get_dataformat(cb_id);
    const InterleavedAddrGenFast<true> d = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write_tile(i, d, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
"""

        def _get_core_coords(core_ranges):
            coords = []
            for cr in core_ranges.ranges():
                for y in range(cr.start.y, cr.end.y + 1):
                    for x in range(cr.start.x, cr.end.x + 1):
                        coords.append(ttnn.CoreCoord(x, y))
            return coords

        def _make_cb_desc(buffer_index, core_ranges, total_size=TILE_SIZE_BF16, is_remote=False, gcb=None):
            cb = ttnn.CBDescriptor()
            cb.total_size = total_size
            cb.core_ranges = core_ranges
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=buffer_index,
                data_format=ttnn.DataType.BFLOAT16,
                page_size=TILE_SIZE_BF16,
            )
            if is_remote:
                cb.remote_format_descriptors = [fmt]
            else:
                cb.format_descriptors = [fmt]
            if gcb is not None:
                cb.set_global_circular_buffer(gcb)
            return cb

        def _make_kernel_desc(source, core_ranges, config, named_ct_args, rt_args_per_core):
            k = ttnn.KernelDescriptor()
            k.kernel_source = source
            k.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            k.core_ranges = core_ranges
            k.named_compile_time_args = named_ct_args
            k.runtime_args = rt_args_per_core
            k.config = config
            return k

        def _build_globalcb_sender_op(input_tensor, core_ranges, gcb, num_tiles):
            src_addr = input_tensor.buffer_address()
            cb_in = _make_cb_desc(0, core_ranges)
            cb_out = _make_cb_desc(4, core_ranges)
            cb_remote = _make_cb_desc(31, core_ranges, total_size=gcb.size(), is_remote=True, gcb=gcb)
            coords = _get_core_coords(core_ranges)
            reader = _make_kernel_desc(
                DRAM_READER_SOURCE,
                core_ranges,
                ttnn.ReaderConfigDescriptor(),
                [("cb_in", 0)],
                [(c, [src_addr, num_tiles]) for c in coords],
            )
            compute = _make_kernel_desc(
                TILE_COPY_COMPUTE_SOURCE,
                core_ranges,
                ttnn.ComputeConfigDescriptor(),
                [("cb_in", 0), ("cb_out", 4)],
                [(c, [num_tiles]) for c in coords],
            )
            writer = _make_kernel_desc(
                GLOBALCB_SENDER_WRITER_SOURCE,
                core_ranges,
                ttnn.WriterConfigDescriptor(),
                [("cb_out", 4), ("cb_remote", 31), ("page_size", TILE_SIZE_BF16)],
                [(c, [num_tiles]) for c in coords],
            )
            desc = ttnn.ProgramDescriptor()
            desc.cbs = [cb_in, cb_out, cb_remote]
            desc.kernels = [reader, compute, writer]
            return OpDescriptor(descriptor=desc, input_tensors=[input_tensor], output_tensors=[], name="gcb_sender")

        def _build_identity_op(input_tensor, output_tensor, core_ranges, num_tiles):
            src_addr = input_tensor.buffer_address()
            dst_addr = output_tensor.buffer_address()
            cb_in = _make_cb_desc(0, core_ranges)
            cb_out = _make_cb_desc(4, core_ranges)
            coords = _get_core_coords(core_ranges)
            reader = _make_kernel_desc(
                DRAM_READER_SOURCE,
                core_ranges,
                ttnn.ReaderConfigDescriptor(),
                [("cb_in", 0)],
                [(c, [src_addr, num_tiles]) for c in coords],
            )
            compute = _make_kernel_desc(
                TILE_COPY_COMPUTE_SOURCE,
                core_ranges,
                ttnn.ComputeConfigDescriptor(),
                [("cb_in", 0), ("cb_out", 4)],
                [(c, [num_tiles]) for c in coords],
            )
            writer = _make_kernel_desc(
                DRAM_WRITER_SOURCE,
                core_ranges,
                ttnn.WriterConfigDescriptor(),
                [("cb_out", 4)],
                [(c, [dst_addr, num_tiles]) for c in coords],
            )
            desc = ttnn.ProgramDescriptor()
            desc.cbs = [cb_in, cb_out]
            desc.kernels = [reader, compute, writer]
            return OpDescriptor(
                descriptor=desc,
                input_tensors=[input_tensor],
                output_tensors=[output_tensor],
                name="identity",
            )

        def _build_globalcb_consumer_op(output_tensor, core_ranges, gcb, num_tiles):
            dst_addr = output_tensor.buffer_address()
            cb_recv = ttnn.CBDescriptor()
            cb_recv.total_size = gcb.size()
            cb_recv.core_ranges = core_ranges
            local_fmt = ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=ttnn.DataType.BFLOAT16, page_size=TILE_SIZE_BF16
            )
            remote_fmt = ttnn.CBFormatDescriptor(
                buffer_index=31, data_format=ttnn.DataType.BFLOAT16, page_size=TILE_SIZE_BF16
            )
            cb_recv.format_descriptors = [local_fmt]
            cb_recv.remote_format_descriptors = [remote_fmt]
            cb_recv.set_global_circular_buffer(gcb)
            coords = _get_core_coords(core_ranges)
            reader = _make_kernel_desc(
                GLOBALCB_RECEIVER_READER_SOURCE,
                core_ranges,
                ttnn.ReaderConfigDescriptor(),
                [("cb_remote", 31), ("cb_in", 0), ("page_size", TILE_SIZE_BF16)],
                [(c, [num_tiles]) for c in coords],
            )
            writer = _make_kernel_desc(
                RECEIVER_DRAM_WRITER_SOURCE,
                core_ranges,
                ttnn.WriterConfigDescriptor(),
                [("cb_in", 0)],
                [(c, [dst_addr, num_tiles]) for c in coords],
            )
            desc = ttnn.ProgramDescriptor()
            desc.cbs = [cb_recv]
            desc.kernels = [reader, writer]
            return OpDescriptor(descriptor=desc, input_tensors=[], output_tensors=[output_tensor], name="gcb_consumer")

        torch.manual_seed(42)
        num_tiles = 8
        shape = [1, 1, 32, 32 * num_tiles]

        torch_input_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_input_b = torch.randn(shape, dtype=torch.bfloat16)

        sender_core = ttnn.CoreCoord(0, 0)
        sender_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        receiver_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

        gcb_size = TILE_SIZE_BF16 * 2
        gcb = ttnn.create_global_circular_buffer(device, [(sender_core, receiver_range)], gcb_size)

        dram = ttnn.DRAM_MEMORY_CONFIG
        tia = ttnn.from_torch(
            torch_input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        tib = ttnn.from_torch(
            torch_input_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        tt_output_b = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        tt_output_recv = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        oa = _build_globalcb_sender_op(tia, sender_range, gcb, num_tiles)
        ob = _build_identity_op(tib, tt_output_b, sender_range, num_tiles)
        con = _build_globalcb_consumer_op(tt_output_recv, receiver_range, gcb, num_tiles)

        fused = [None]

        def build_and_launch():
            fused[0] = Parallel(Sequential(oa, ob), con).build(device)
            fused[0].launch()

        cold, warm = _time_cold_warm(build_and_launch, device, warm_fn=lambda: fused[0].launch())

        result_recv = ttnn.to_torch(tt_output_recv)
        result_b = ttnn.to_torch(tt_output_b)

        passing_recv, pcc_recv = comp_pcc(torch_input_a, result_recv, pcc=0.999)
        passing_b, pcc_b = comp_pcc(torch_input_b, result_b, pcc=0.999)

        print(f"\n  Demo 5 Fused: cold={cold:.2f}ms  warm={warm:.2f}ms  PCC: recv={pcc_recv:.4f} phase1={pcc_b:.4f}")
        assert passing_recv, f"Receiver PCC: {pcc_recv}"
        assert passing_b, f"Phase 1 PCC: {pcc_b}"

    # =================================================================
    # Demo 8: Sharded Heterogeneous Tree
    # =================================================================
    #
    # Same balanced binary tree topology as demo 7, but with block-sharded
    # intermediates instead of interleaved DRAM I/O.  Scaled down to fit
    # height-sharded in L1.
    #
    # Stem input is height-sharded across 16 cores for a fair comparison:
    # both fused and unfused paths use the same core grids per op.
    #
    #   Level 0:  LN_stem            (16 cores)  [1,1,2048,256]  (sharded)
    #   Level 1:  Slice → MM_left    ( 8 cores)  [1,1,1024,256] × B_left → [1,1,1024,128]
    #             Slice → MM_right   ( 8 cores)  [1,1,1024,256] × B_right → [1,1,1024,128]
    #   Level 2:  Slice → LN_ll/lr   ( 4 cores)  [1,1,512,128]
    #             Slice → LN_rl/rr   ( 4 cores)  [1,1,512,128]

    def _demo8_setup(self, device):
        torch.manual_seed(42)
        # Scaled down to fit block-sharded in L1.  Original was [8192, 1024].
        rows, cols = 2048, 256
        mm_n = 128  # matmul output width (B = [256, 128] = 64KB)

        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})
        left_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})
        right_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))})
        ll_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        lr_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 7))})
        rl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3))})
        rr_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 4), ttnn.CoreCoord(1, 7))})

        # Matmul config for 8 cores (1x8 grid):
        # A=[1,1,1024,256] → M=32 tiles, K=8 tiles
        # B=[1,1,256,128] → output [1,1,1024,128]
        # in0_block_w must equal shard_w / tile_w for block-sharded A input
        # (shard [128,256] on 1-col grid → shard_w=256, so in0_block_w=256/32=8)
        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 8),
            in0_block_w=cols // 32,
            out_subblock_h=1,
            out_subblock_w=min(mm_n // 32, 4),
            per_core_M=4,
            per_core_N=mm_n // 32,
        )

        # Block-sharded: height ÷ grid_rows, width ÷ grid_cols.
        def _shard_mem(cores, shard_h, shard_w):
            spec = ttnn.ShardSpec(cores, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        # All intermediate shard configs, keyed by position in the tree.
        # stem [2048,256] on 2×8 → [256,128]; left/right [1024,256] on 1×8 → [128,256]
        # mm [1024,128] on 1×8 → [128,128]; leaf [512,128] on 1×4 → [128,128]
        shards = {
            "stem": _shard_mem(stem_cores, 256, 128),  # [256,128] × 16 cores
            "left": _shard_mem(left_cores, 128, 256),  # [128,256] × 8 cores
            "right": _shard_mem(right_cores, 128, 256),  # [128,256] × 8 cores
            "mm_left": _shard_mem(left_cores, 128, 128),  # [128,128] × 8 cores
            "mm_right": _shard_mem(right_cores, 128, 128),  # [128,128] × 8 cores
            "ll": _shard_mem(ll_cores, 128, 128),  # [128,128] × 4 cores
            "lr": _shard_mem(lr_cores, 128, 128),  # [128,128] × 4 cores
            "rl": _shard_mem(rl_cores, 128, 128),  # [128,128] × 4 cores
            "rr": _shard_mem(rr_cores, 128, 128),  # [128,128] × 4 cores
        }

        torch_input = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=shards["stem"],
        )

        # Matmul B [1,1,256,128] must be interleaved for MatmulMultiCoreReuseProgramConfig
        dram = ttnn.DRAM_MEMORY_CONFIG
        tt_B_left = ttnn.from_torch(
            torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        tt_B_right = ttnn.from_torch(
            torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )

        return (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
            shards,
        )

    def _demo8_make_ops(self, device):
        """Create all OpDescriptors for Demo 8's sharded tree."""
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice_op
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
            shards,
        ) = self._demo8_setup(device)

        rows, cols = 2048, 256
        half = rows // 2
        quarter = rows // 4

        # Level 0: stem LN on 16 cores (auto-detects sharded input grid)
        ln_stem = layer_norm.layer_norm(
            tt_input, core_range_set=stem_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )

        # Level 1: slice (sharded on 8 cores) → matmul (sharded output on 8 cores)
        sl_top = slice_op(
            ln_stem.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, half, cols],
            core_range_set=left_cores,
            memory_config=shards["left"],
        )
        sl_bot = slice_op(
            ln_stem.output_tensors[0],
            [0, 0, half, 0],
            [1, 1, rows, cols],
            core_range_set=right_cores,
            memory_config=shards["right"],
        )

        mm_left = matmul_desc(
            sl_top.output_tensors[0],
            tt_B_left,
            core_range_set=left_cores,
            program_config=mm_cfg,
            compute_kernel_config=COMPUTE_CONFIG,
            output_mem_config=shards["mm_left"],
        )
        mm_right = matmul_desc(
            sl_bot.output_tensors[0],
            tt_B_right,
            core_range_set=right_cores,
            program_config=mm_cfg,
            compute_kernel_config=COMPUTE_CONFIG,
            output_mem_config=shards["mm_right"],
        )

        # Level 2: slice (sharded on 4 cores) → LN (auto-detects from sharded input)
        sl_tl = slice_op(
            mm_left.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, quarter, mm_n],
            core_range_set=ll_cores,
            memory_config=shards["ll"],
        )
        sl_bl = slice_op(
            mm_left.output_tensors[0],
            [0, 0, quarter, 0],
            [1, 1, half, mm_n],
            core_range_set=lr_cores,
            memory_config=shards["lr"],
        )
        sl_tr = slice_op(
            mm_right.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, quarter, mm_n],
            core_range_set=rl_cores,
            memory_config=shards["rl"],
        )
        sl_br = slice_op(
            mm_right.output_tensors[0],
            [0, 0, quarter, 0],
            [1, 1, half, mm_n],
            core_range_set=rr_cores,
            memory_config=shards["rr"],
        )

        ln_ll = layer_norm.layer_norm(
            sl_tl.output_tensors[0], core_range_set=ll_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )
        ln_lr = layer_norm.layer_norm(
            sl_bl.output_tensors[0], core_range_set=lr_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )
        ln_rl = layer_norm.layer_norm(
            sl_tr.output_tensors[0], core_range_set=rl_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )
        ln_rr = layer_norm.layer_norm(
            sl_br.output_tensors[0], core_range_set=rr_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )

        return (
            ln_stem,
            sl_top,
            sl_bot,
            mm_left,
            mm_right,
            sl_tl,
            sl_bl,
            sl_tr,
            sl_br,
            ln_ll,
            ln_lr,
            ln_rl,
            ln_rr,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
            shards,
        )

    def _demo8_build_fused(self, device, ops):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr) = ops
        return Sequential(
            ln_stem,
            Parallel(
                Sequential(sl_top, mm_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
                Sequential(sl_bot, mm_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
            ),
        ).build(device)

    def test_demo8_fused(self, device):
        (
            ln_stem,
            sl_top,
            sl_bot,
            mm_left,
            mm_right,
            sl_tl,
            sl_bl,
            sl_tr,
            sl_br,
            ln_ll,
            ln_lr,
            ln_rl,
            ln_rr,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
            shards,
        ) = self._demo8_make_ops(device)

        rows, cols = 2048, 256
        half = rows // 2
        quarter = rows // 4
        ops = (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr)

        if self._SINGLE_RUN_ONLY:
            fused = self._demo8_build_fused(device, ops)
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Demo 8 Fused: single run (for Tracy)")
        else:
            # Cold start
            _, cold, _ = _time_fused(lambda: self._demo8_build_fused(device, ops), device)

            # Steady-state e2e
            fused = self._demo8_build_fused(device, ops)
            e2e = _time_steady_state(fused.launch, device)

            # Unfused reference for PCC — sharded intermediates on (0,0)-based grids.
            # Right path reuses left-path shard configs (same shapes, serialized).
            stem_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(2, 8),
                subblock_w=min(128 // 32, 4),
                block_h=256 // 32,
                block_w=128 // 32,
                inplace=False,
            )
            leaf_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(1, 4),
                subblock_w=min(128 // 32, 4),
                block_h=128 // 32,
                block_w=128 // 32,
                inplace=False,
            )
            u_stem = ttnn.layer_norm(
                tt_input,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=stem_ln_cfg,
                memory_config=shards["stem"],
            )

            # Left path: stem → slice top → matmul → slice top-left → LN
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=shards["left"])
            # Right path slice before deallocating stem
            u_bot = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols], memory_config=shards["left"])
            ttnn.deallocate(u_stem)

            u_left = ttnn.matmul(
                u_top,
                tt_B_left,
                program_config=mm_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=shards["mm_left"],
            )
            ttnn.deallocate(u_top)
            u_tl = ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, mm_n], memory_config=shards["ll"])
            ttnn.deallocate(u_left)
            ref_ll = ttnn.to_torch(
                ttnn.layer_norm(
                    u_tl,
                    epsilon=1e-5,
                    compute_kernel_config=COMPUTE_CONFIG,
                    program_config=leaf_ln_cfg,
                    memory_config=shards["ll"],
                )
            )
            ttnn.deallocate(u_tl)
            result_ll = ttnn.to_torch(ln_ll.output_tensors[0])

            # Right path (reuses (0,0)-based left-path shard configs)
            u_right = ttnn.matmul(
                u_bot,
                tt_B_right,
                program_config=mm_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=shards["mm_left"],
            )
            ttnn.deallocate(u_bot)
            u_tr = ttnn.slice(u_right, [0, 0, 0, 0], [1, 1, quarter, mm_n], memory_config=shards["ll"])
            ttnn.deallocate(u_right)
            ref_rl = ttnn.to_torch(
                ttnn.layer_norm(
                    u_tr,
                    epsilon=1e-5,
                    compute_kernel_config=COMPUTE_CONFIG,
                    program_config=leaf_ln_cfg,
                    memory_config=shards["ll"],
                )
            )
            ttnn.deallocate(u_tr)
            result_rl = ttnn.to_torch(ln_rl.output_tensors[0])

            p_ll, pcc_ll = comp_pcc(ref_ll, result_ll, pcc=0.97)
            p_rl, pcc_rl = comp_pcc(ref_rl, result_rl, pcc=0.97)
            print(f"\n  Demo 8 Fused: cold={cold:.2f}ms  e2e={e2e:.3f}ms  PCC: ll={pcc_ll:.6f} rl={pcc_rl:.6f}")
            assert p_ll, f"Left-left PCC: {pcc_ll}"
            assert p_rl, f"Right-left PCC: {pcc_rl}"

    def test_demo8_unfused(self, device):
        """Unfused path using ttnn ops with sharded intermediates.

        ttnn ops require (0,0)-based core grids (compute_with_storage_grid_size
        is always (0,0)-origin), so we can't use the fused path's non-origin
        grids (e.g. right_cores=(1,0)-(1,7)). Instead, all ops use (0,0)-based
        grids with matching shard shapes. This means left/right paths can't run
        on disjoint cores — they serialize, which is the normal unfused behavior.

        ttnn.slice operates directly on sharded TILE inputs via the tile factory
        (which uses TensorAccessor to handle both interleaved and sharded buffers).
        """
        torch.manual_seed(42)
        rows, cols = 2048, 256
        mm_n = 128
        half = rows // 2
        quarter = rows // 4

        def _shard_mem(cores, shard_h, shard_w):
            spec = ttnn.ShardSpec(cores, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        # All (0,0)-based grids
        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})
        branch_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})
        leaf_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})

        stem_mem = _shard_mem(stem_cores, 256, 128)
        branch_mem = _shard_mem(branch_cores, 128, 256)
        mm_mem = _shard_mem(branch_cores, 128, 128)
        leaf_mem = _shard_mem(leaf_cores, 128, 128)

        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 8),
            in0_block_w=cols // 32,
            out_subblock_h=1,
            out_subblock_w=min(mm_n // 32, 4),
            per_core_M=4,
            per_core_N=mm_n // 32,
        )

        ln_prog_cfg = lambda gx, gy, bh, bw: ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=min(bw, 4),
            block_h=bh,
            block_w=bw,
            inplace=False,
        )
        stem_ln_cfg = ln_prog_cfg(2, 8, 256 // 32, 128 // 32)
        leaf_ln_cfg = ln_prog_cfg(1, 4, 128 // 32, 128 // 32)

        tt_input = ttnn.from_torch(
            torch.randn(1, 1, rows, cols, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=stem_mem,
        )
        # Matmul B [1,1,256,128] must be interleaved for MatmulMultiCoreReuseProgramConfig
        dram = ttnn.DRAM_MEMORY_CONFIG
        tt_B_left = ttnn.from_torch(
            torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        tt_B_right = ttnn.from_torch(
            torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )

        def unfused():
            u_stem = ttnn.layer_norm(
                tt_input,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=stem_ln_cfg,
                memory_config=stem_mem,
            )
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=branch_mem)
            u_bot = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols], memory_config=branch_mem)
            u_left = ttnn.matmul(
                u_top, tt_B_left, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG, memory_config=mm_mem
            )
            u_right = ttnn.matmul(
                u_bot, tt_B_right, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG, memory_config=mm_mem
            )
            u_tl = ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, mm_n], memory_config=leaf_mem)
            u_bl = ttnn.slice(u_left, [0, 0, quarter, 0], [1, 1, half, mm_n], memory_config=leaf_mem)
            u_tr = ttnn.slice(u_right, [0, 0, 0, 0], [1, 1, quarter, mm_n], memory_config=leaf_mem)
            u_br = ttnn.slice(u_right, [0, 0, quarter, 0], [1, 1, half, mm_n], memory_config=leaf_mem)
            ttnn.layer_norm(
                u_tl,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=leaf_ln_cfg,
                memory_config=leaf_mem,
            )
            ttnn.layer_norm(
                u_bl,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=leaf_ln_cfg,
                memory_config=leaf_mem,
            )
            ttnn.layer_norm(
                u_tr,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=leaf_ln_cfg,
                memory_config=leaf_mem,
            )
            ttnn.layer_norm(
                u_br,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=leaf_ln_cfg,
                memory_config=leaf_mem,
            )

        if self._SINGLE_RUN_ONLY:
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Demo 8 Unfused: single run (for Tracy)")
        else:
            # Cold start
            cold, _ = _time_cold_warm(unfused, device)

            # Steady-state e2e
            e2e = _time_steady_state(unfused, device)

            print(f"\n  Demo 8 Unfused: cold={cold:.2f}ms  e2e={e2e:.3f}ms")
