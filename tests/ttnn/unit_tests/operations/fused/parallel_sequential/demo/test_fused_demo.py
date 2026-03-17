# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion Infrastructure Demo Suite

Demos showcasing different fusion capabilities:

Performance demos (fused vs unfused comparison):
- Linear Chain: RMS -> Matmul -> RMS (DRAM interleaved, 4x2 grid)
- Sharded Chain: RMS -> LN (block-sharded, 4x4 grid)
- Parallel Chains: LN->MM + RMS->MM on disjoint 1x8 core columns
- Sharded Tree: LN -> Slice -> Matmul -> Slice -> LN (5-level binary tree, 2x8 grid)
- Asymmetric Branches: LN stem -> Parallel(Slice->RMS->RMS, Slice->LN) (4x8 grid)

Functional demos (fused only):
- GlobalCircularBuffer: mid-kernel write to external consumer
- Non-Contiguous Grid: identity chain with branching on scattered cores (unicast barrier)
- Barrier Overhead: N no-op phases measuring pure barrier mechanism cost

Each perf demo is split into separate fused and unfused tests.
Cold = all caches cleared (JIT disk + in-memory + program + fusion build).

Each test is parametrized by perf_mode: "cold_start", "e2e", or "device_fw".
Run subsets: pytest ... -k cold_start  (or -k e2e, or -k device_fw)
For Tracy: export TT_METAL_DEVICE_PROFILER=1 && python -m tracy -r -m pytest ... -k device_fw
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

# No-op kernels: empty kernel_main(), used for pure barrier overhead measurement.
NOOP_READER_SOURCE = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
NOOP_COMPUTE_SOURCE = (
    '#include "api/compute/compute_kernel_api.h"\n' '#include "api/compute/common.h"\n' "void kernel_main() {}\n"
)
NOOP_WRITER_SOURCE = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'


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


def _build_noop_op(core_ranges, dummy_tensor):
    """Build a no-op OpDescriptor: 3 empty kernels, no CBs.

    All three RISCs enter kernel_main() and immediately return.
    A dummy tensor is passed through to satisfy the generic_op non-empty
    tensor assertion — the kernels never touch it.
    """
    coords = _get_core_coords(core_ranges)
    # Dummy RT arg (unused) — RuntimeArgsView requires ≥1 arg per core to register the entry.
    dummy = [(c, [0]) for c in coords]
    reader = _make_kernel_desc(NOOP_READER_SOURCE, core_ranges, ttnn.ReaderConfigDescriptor(), [], dummy)
    compute = _make_kernel_desc(NOOP_COMPUTE_SOURCE, core_ranges, ttnn.ComputeConfigDescriptor(), [], dummy)
    writer = _make_kernel_desc(NOOP_WRITER_SOURCE, core_ranges, ttnn.WriterConfigDescriptor(), [], dummy)
    desc = ttnn.ProgramDescriptor()
    desc.cbs = []
    desc.kernels = [reader, compute, writer]
    return OpDescriptor(descriptor=desc, input_tensors=[dummy_tensor], output_tensors=[dummy_tensor], name="noop")


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
    for kernels_dir in cache_dir.glob("*/kernels"):
        shutil.rmtree(kernels_dir)


def _time_cold(fn, device):
    """Clear all caches, run fn() once, return cold_ms."""
    sync = lambda: ttnn.synchronize_device(device)
    _clear_all_caches(device)
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return 1000 * (time.perf_counter() - t0)


def _time_cold_fused(build_fn, device):
    """Clear all caches, build + launch a fused op from cold. Returns cold_ms."""
    return _time_cold(lambda: build_fn().launch(), device)


def _time_e2e(fn, device, num_warmup=5, num_measure=100):
    """Measure steady-state E2E time per iteration (ms).

    Runs num_warmup iterations (discarded), then num_measure iterations
    timed as a batch, returning total_ms / num_measure.
    All caches are warm (program cache, build cache, JIT cache).
    """
    sync = lambda: ttnn.synchronize_device(device)

    for _ in range(num_warmup):
        fn()
    sync()

    t0 = time.perf_counter()
    for _ in range(num_measure):
        fn()
    sync()
    return 1000 * (time.perf_counter() - t0) / num_measure


# =============================================================================
# Performance Demos (fused vs unfused comparison)
# =============================================================================


class TestPerfDemos:
    """Performance comparison demos: fused vs unfused with timing and PCC.

    Each demo has separate fused and unfused tests with cold start and
    steady-state E2E timing.
    """

    # -----------------------------------------------------------------
    # Linear Chain — RMS -> Matmul -> RMS (DRAM, 4x2 grid)
    # Input [256, H], matmul [H, H], 4x2 = 8 cores
    # H must be divisible by 128 (in0_block_w=4 x tile=32)
    # -----------------------------------------------------------------

    def _linear_chain_setup(self, device, H):
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

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_linear_chain_rms_matmul_rms_fused(self, device, H, perf_mode):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        core_range, mm_cfg, torch_input, torch_w, torch_b = self._linear_chain_setup(device, H)

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

        if perf_mode == "device_fw":
            fused = Sequential(r1, m, r2).build()
            fused.launch()
            ttnn.synchronize_device(device)
            print(f"\n  Linear Chain Fused (H={H}): device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold_fused(lambda: Sequential(r1, m, r2).build(), device)

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
            print(f"\n  Linear Chain Fused (H={H}): cold={cold:.2f}ms PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"
        elif perf_mode == "e2e":
            fused = Sequential(r1, m, r2).build()
            e2e = _time_e2e(fused.launch, device)

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
            print(f"\n  Linear Chain Fused (H={H}): e2e={e2e:.3f}ms PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_linear_chain_rms_matmul_rms_unfused(self, device, H, perf_mode):
        core_range, mm_cfg, torch_input, torch_w, torch_b = self._linear_chain_setup(device, H)
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

        if perf_mode == "device_fw":
            unfused()
            ttnn.synchronize_device(device)
            print(f"\n  Linear Chain Unfused (H={H}): device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold(unfused, device)
            print(f"\n  Linear Chain Unfused (H={H}): cold={cold:.2f}ms")
        elif perf_mode == "e2e":
            e2e = _time_e2e(unfused, device)
            print(f"\n  Linear Chain Unfused (H={H}): e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Sharded Chain — RMS -> LN (block-sharded, 4x4 grid)
    # Input [H, 512] on 4x4=16 cores, shard [H/4, 128]
    # H must be divisible by 128 (4 grid-rows x tile=32)
    # -----------------------------------------------------------------

    def _sharded_chain_setup(self, device, H):
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
        # Norm weight [1,1,1,512] width-sharded across 4 columns -> [32,128] per core
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

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_sharded_chain_rms_layernorm_fused(self, device, H, perf_mode):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm

        cores, sharded_mem, program_cfg, tt_input, tt_w = self._sharded_chain_setup(device, H)

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

        if perf_mode == "device_fw":
            fused = Sequential(r, ln).build()
            fused.launch()
            ttnn.synchronize_device(device)
            print(f"\n  Sharded Chain Fused (H={H}): device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold_fused(lambda: Sequential(r, ln).build(), device)

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
            print(f"\n  Sharded Chain Fused (H={H}): cold={cold:.2f}ms PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"
        elif perf_mode == "e2e":
            fused = Sequential(r, ln).build()
            e2e = _time_e2e(fused.launch, device)

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
            print(f"\n  Sharded Chain Fused (H={H}): e2e={e2e:.3f}ms PCC={pcc:.6f}")
            assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    @pytest.mark.parametrize("H", [128, 1536], ids=["H128", "H1536"])
    def test_sharded_chain_rms_layernorm_unfused(self, device, H, perf_mode):
        cores, sharded_mem, program_cfg, tt_input, tt_w = self._sharded_chain_setup(device, H)

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

        if perf_mode == "device_fw":
            unfused()
            ttnn.synchronize_device(device)
            print(f"\n  Sharded Chain Unfused (H={H}): device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold(unfused, device)
            print(f"\n  Sharded Chain Unfused (H={H}): cold={cold:.2f}ms")
        elif perf_mode == "e2e":
            e2e = _time_e2e(unfused, device)
            print(f"\n  Sharded Chain Unfused (H={H}): e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Parallel Chains — LN->MM + RMS->MM on disjoint 1x8 columns
    # Chain A: LN -> Matmul on cores col 0, rows 0-7
    # Chain B: RMS -> Matmul on cores col 1, rows 0-7
    # Block-sharded [1024,256] inputs, sharded [1024,128] outputs on
    # 1x8 grid. B weight + norm weight/bias L1 sharded.
    # -----------------------------------------------------------------

    def _parallel_chains_setup(self, device):
        torch.manual_seed(42)
        rows, K, N = 1024, 256, 128

        # 1-column grids: K (width) not split -> matmul factory CB fits in shard
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

        # [1024,256] x [256,128] on 1x8 grid
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

        # Norm weight/bias [1,1,1,256] width-sharded across 8 cores -> [32,32] per core
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

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_parallel_chains_ln_mm_rms_mm_fused(self, device, perf_mode):
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
        ) = self._parallel_chains_setup(device)

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

        if perf_mode == "device_fw":
            fused = Parallel(Sequential(la, ma), Sequential(rb, mb)).build()
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Parallel Chains Fused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold_fused(lambda: Parallel(Sequential(la, ma), Sequential(rb, mb)).build(), device)

            result_a = ttnn.to_torch(ma.output_tensors[0])
            result_b = ttnn.to_torch(mb.output_tensors[0])

            # Unfused reference for PCC — interleaved to avoid core mapping constraints
            ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ua2 = ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)
            ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ub2 = ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)

            p_a, pcc_a = comp_pcc(ttnn.to_torch(ua2), result_a, pcc=0.97)
            p_b, pcc_b = comp_pcc(ttnn.to_torch(ub2), result_b, pcc=0.97)

            print(f"\n  Parallel Chains Fused: cold={cold:.2f}ms PCC: a={pcc_a:.4f} b={pcc_b:.4f}")
            assert p_a, f"Chain A PCC: {pcc_a}"
            assert p_b, f"Chain B PCC: {pcc_b}"
        elif perf_mode == "e2e":
            fused = Parallel(Sequential(la, ma), Sequential(rb, mb)).build()
            e2e = _time_e2e(fused.launch, device)

            result_a = ttnn.to_torch(ma.output_tensors[0])
            result_b = ttnn.to_torch(mb.output_tensors[0])

            # Unfused reference for PCC — interleaved to avoid core mapping constraints
            ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ua2 = ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)
            ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG)
            ub2 = ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=COMPUTE_CONFIG)

            p_a, pcc_a = comp_pcc(ttnn.to_torch(ua2), result_a, pcc=0.97)
            p_b, pcc_b = comp_pcc(ttnn.to_torch(ub2), result_b, pcc=0.97)

            print(f"\n  Parallel Chains Fused: e2e={e2e:.3f}ms PCC: a={pcc_a:.4f} b={pcc_b:.4f}")
            assert p_a, f"Chain A PCC: {pcc_a}"
            assert p_b, f"Chain B PCC: {pcc_b}"

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_parallel_chains_ln_mm_rms_mm_unfused(self, device, perf_mode):
        """Unfused path using ttnn ops with sharded intermediates.

        Both chains serialize on the same (0,0)-based 1x8 grid with matching
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
        # Norm weight/bias [1,1,1,256] width-sharded across 8 cores -> [32,32] per core
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

        if perf_mode == "device_fw":
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Parallel Chains Unfused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold(unfused, device)
            print(f"\n  Parallel Chains Unfused: cold={cold:.2f}ms")
        elif perf_mode == "e2e":
            e2e = _time_e2e(unfused, device)
            print(f"\n  Parallel Chains Unfused: e2e={e2e:.3f}ms")

    # =================================================================
    # Sharded Tree — LN -> Slice -> Matmul -> Slice -> LN
    # =================================================================
    #
    # Balanced binary tree topology with block-sharded
    # intermediates instead of interleaved DRAM I/O.  Scaled down to fit
    # height-sharded in L1.
    #
    # Stem input is height-sharded across 16 cores for a fair comparison:
    # both fused and unfused paths use the same core grids per op.
    #
    #   Level 0:  LN_stem            (16 cores)  [1,1,2048,256]  (sharded)
    #   Level 1:  Slice -> MM_left    ( 8 cores)  [1,1,1024,256] x B_left -> [1,1,1024,128]
    #             Slice -> MM_right   ( 8 cores)  [1,1,1024,256] x B_right -> [1,1,1024,128]
    #   Level 2:  Slice -> LN_ll/lr   ( 4 cores)  [1,1,512,128]
    #             Slice -> LN_rl/rr   ( 4 cores)  [1,1,512,128]

    def _sharded_tree_setup(self, device):
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
        # A=[1,1,1024,256] -> M=32 tiles, K=8 tiles
        # B=[1,1,256,128] -> output [1,1,1024,128]
        # in0_block_w must equal shard_w / tile_w for block-sharded A input
        # (shard [128,256] on 1-col grid -> shard_w=256, so in0_block_w=256/32=8)
        mm_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 8),
            in0_block_w=cols // 32,
            out_subblock_h=1,
            out_subblock_w=min(mm_n // 32, 4),
            per_core_M=4,
            per_core_N=mm_n // 32,
        )

        # Block-sharded: height / grid_rows, width / grid_cols.
        def _shard_mem(cores, shard_h, shard_w):
            spec = ttnn.ShardSpec(cores, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        # All intermediate shard configs, keyed by position in the tree.
        # stem [2048,256] on 2x8 -> [256,128]; left/right [1024,256] on 1x8 -> [128,256]
        # mm [1024,128] on 1x8 -> [128,128]; leaf [512,128] on 1x4 -> [128,128]
        shards = {
            "stem": _shard_mem(stem_cores, 256, 128),  # [256,128] x 16 cores
            "left": _shard_mem(left_cores, 128, 256),  # [128,256] x 8 cores
            "right": _shard_mem(right_cores, 128, 256),  # [128,256] x 8 cores
            "mm_left": _shard_mem(left_cores, 128, 128),  # [128,128] x 8 cores
            "mm_right": _shard_mem(right_cores, 128, 128),  # [128,128] x 8 cores
            "ll": _shard_mem(ll_cores, 128, 128),  # [128,128] x 4 cores
            "lr": _shard_mem(lr_cores, 128, 128),  # [128,128] x 4 cores
            "rl": _shard_mem(rl_cores, 128, 128),  # [128,128] x 4 cores
            "rr": _shard_mem(rr_cores, 128, 128),  # [128,128] x 4 cores
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

    def _sharded_tree_make_ops(self, device):
        """Create all OpDescriptors for the sharded tree."""
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice
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
        ) = self._sharded_tree_setup(device)

        rows, cols = 2048, 256
        half = rows // 2
        quarter = rows // 4

        # Level 0: stem LN on 16 cores (auto-detects sharded input grid)
        ln_stem = layer_norm.layer_norm(
            tt_input, core_range_set=stem_cores, epsilon=1e-5, compute_kernel_config=COMPUTE_CONFIG
        )

        # Level 1: slice (sharded on 8 cores) -> matmul (sharded output on 8 cores)
        sl_top = slice(
            ln_stem.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, half, cols],
            core_range_set=left_cores,
            memory_config=shards["left"],
        )
        sl_bot = slice(
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

        # Level 2: slice (sharded on 4 cores) -> LN (auto-detects from sharded input)
        sl_tl = slice(
            mm_left.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, quarter, mm_n],
            core_range_set=ll_cores,
            memory_config=shards["ll"],
        )
        sl_bl = slice(
            mm_left.output_tensors[0],
            [0, 0, quarter, 0],
            [1, 1, half, mm_n],
            core_range_set=lr_cores,
            memory_config=shards["lr"],
        )
        sl_tr = slice(
            mm_right.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, quarter, mm_n],
            core_range_set=rl_cores,
            memory_config=shards["rl"],
        )
        sl_br = slice(
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

    def _sharded_tree_build_fused(self, device, ops):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr) = ops
        return Sequential(
            ln_stem,
            Parallel(
                Sequential(sl_top, mm_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
                Sequential(sl_bot, mm_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
            ),
        ).build()

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_sharded_tree_ln_slice_matmul_slice_ln_fused(self, device, perf_mode):
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
        ) = self._sharded_tree_make_ops(device)

        rows, cols = 2048, 256
        half = rows // 2
        quarter = rows // 4
        ops = (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr)

        if perf_mode == "device_fw":
            fused = self._sharded_tree_build_fused(device, ops)
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Sharded Tree Fused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold_fused(lambda: self._sharded_tree_build_fused(device, ops), device)

            # Unfused reference for PCC — sharded intermediates on (0,0)-based grids.
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
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=shards["left"])
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
            print(f"\n  Sharded Tree Fused: cold={cold:.2f}ms PCC: ll={pcc_ll:.6f} rl={pcc_rl:.6f}")
            assert p_ll, f"Left-left PCC: {pcc_ll}"
            assert p_rl, f"Right-left PCC: {pcc_rl}"
        elif perf_mode == "e2e":
            fused = self._sharded_tree_build_fused(device, ops)
            e2e = _time_e2e(fused.launch, device)

            # Unfused reference for PCC — sharded intermediates on (0,0)-based grids.
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
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=shards["left"])
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
            print(f"\n  Sharded Tree Fused: e2e={e2e:.3f}ms PCC: ll={pcc_ll:.6f} rl={pcc_rl:.6f}")
            assert p_ll, f"Left-left PCC: {pcc_ll}"
            assert p_rl, f"Right-left PCC: {pcc_rl}"

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_sharded_tree_ln_slice_matmul_slice_ln_unfused(self, device, perf_mode):
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

        if perf_mode == "device_fw":
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Sharded Tree Unfused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold(unfused, device)
            print(f"\n  Sharded Tree Unfused: cold={cold:.2f}ms")
        elif perf_mode == "e2e":
            e2e = _time_e2e(unfused, device)
            print(f"\n  Sharded Tree Unfused: e2e={e2e:.3f}ms")

    # -----------------------------------------------------------------
    # Asymmetric Branches — LN stem -> Parallel(Slice->RMS->RMS, Slice->LN)
    #
    # Demonstrates the benefit of parallel + sequential fusion when one
    # branch is a chain of lightweight ops running alongside a single
    # heavy compute op on disjoint cores.  Uses 32-core stem (4x8) and
    # 16-core branches (2x8 each).
    # -----------------------------------------------------------------

    def _asymmetric_branches_setup(self, device):
        torch.manual_seed(42)
        rows, cols = 2048, 512

        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))})
        left_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})
        right_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 7))})

        def _shard_mem(cores, shard_h, shard_w):
            spec = ttnn.ShardSpec(cores, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        shards = {
            "stem": _shard_mem(stem_cores, 256, 128),  # [2048,512] on 4x8
            "left": _shard_mem(left_cores, 128, 256),  # [1024,512] on 2x8
            "right": _shard_mem(right_cores, 128, 256),  # [1024,512] on 2x8
        }

        torch_input = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=shards["stem"],
        )

        # RMS weight [1,1,1,512] width-sharded on left cores -> [32,32] per core
        torch_w = torch.ones(1, 1, 1, cols, dtype=torch.bfloat16)
        w_shard = ttnn.ShardSpec(left_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
        tw = ttnn.from_torch(torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem)

        return (
            stem_cores,
            left_cores,
            right_cores,
            shards,
            tt_input,
            tw,
        )

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_asymmetric_branches_ln_slice_rms_ln_fused(self, device, perf_mode):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice

        (
            stem_cores,
            left_cores,
            right_cores,
            shards,
            tt_input,
            tw,
        ) = self._asymmetric_branches_setup(device)
        rows, cols = 2048, 512
        half = rows // 2

        # Stem: LN on 32 cores
        ln_stem = layer_norm.layer_norm(
            tt_input,
            core_range_set=stem_cores,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
        )

        # Split to disjoint branches
        sl_left = slice(
            ln_stem.output_tensors[0],
            [0, 0, 0, 0],
            [1, 1, half, cols],
            core_range_set=left_cores,
            memory_config=shards["left"],
        )
        sl_right = slice(
            ln_stem.output_tensors[0],
            [0, 0, half, 0],
            [1, 1, rows, cols],
            core_range_set=right_cores,
            memory_config=shards["right"],
        )

        # Left branch: chain of 2 RMS norms
        rms1 = rms_norm.rms_norm(
            sl_left.output_tensors[0],
            core_range_set=left_cores,
            weight=tw,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=shards["left"],
        )
        rms2 = rms_norm.rms_norm(
            rms1.output_tensors[0],
            core_range_set=left_cores,
            weight=tw,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=shards["left"],
        )

        # Right branch: single LN (heavy compute)
        ln_right = layer_norm.layer_norm(
            sl_right.output_tensors[0],
            core_range_set=right_cores,
            epsilon=1e-5,
            compute_kernel_config=COMPUTE_CONFIG,
        )

        def build():
            return Sequential(
                ln_stem,
                Parallel(
                    Sequential(sl_left, rms1, rms2),
                    Sequential(sl_right, ln_right),
                ),
            ).build()

        def _pcc_check():
            # Unfused reference for PCC — sharded on (0,0)-based grids
            branch_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})
            stem_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(4, 8),
                subblock_w=min(128 // 32, 4),
                block_h=256 // 32,
                block_w=128 // 32,
                inplace=False,
            )
            branch_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(2, 8),
                subblock_w=min(256 // 32, 4),
                block_h=128 // 32,
                block_w=256 // 32,
                inplace=False,
            )
            branch_mem = shards["left"]  # (0,0)-based 2x8

            u_stem = ttnn.layer_norm(
                tt_input,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=stem_ln_cfg,
                memory_config=shards["stem"],
            )
            u_left = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=branch_mem)
            u_right = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols], memory_config=branch_mem)
            ttnn.deallocate(u_stem)

            for _ in range(2):
                u_left = ttnn.rms_norm(
                    u_left,
                    weight=tw,
                    epsilon=1e-5,
                    program_config=branch_ln_cfg,
                    compute_kernel_config=COMPUTE_CONFIG,
                    memory_config=branch_mem,
                )
            u_right = ttnn.layer_norm(
                u_right,
                epsilon=1e-5,
                program_config=branch_ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=branch_mem,
            )

            result_left = ttnn.to_torch(rms2.output_tensors[0])
            result_right = ttnn.to_torch(ln_right.output_tensors[0])
            ref_left = ttnn.to_torch(u_left)
            ref_right = ttnn.to_torch(u_right)

            p_l, pcc_l = comp_pcc(ref_left, result_left, pcc=0.97)
            p_r, pcc_r = comp_pcc(ref_right, result_right, pcc=0.97)
            assert p_l, f"Left chain PCC: {pcc_l}"
            assert p_r, f"Right LN PCC: {pcc_r}"
            return pcc_l, pcc_r

        if perf_mode == "device_fw":
            fused = build()
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Asymmetric Branches Fused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold_fused(build, device)
            pcc_l, pcc_r = _pcc_check()
            print(f"\n  Asymmetric Branches Fused: cold={cold:.2f}ms PCC: left={pcc_l:.4f} right={pcc_r:.4f}")
        elif perf_mode == "e2e":
            fused = build()
            e2e = _time_e2e(fused.launch, device)
            pcc_l, pcc_r = _pcc_check()
            print(f"\n  Asymmetric Branches Fused: e2e={e2e:.3f}ms PCC: left={pcc_l:.4f} right={pcc_r:.4f}")

    @pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
    def test_asymmetric_branches_ln_slice_rms_ln_unfused(self, device, perf_mode):
        """Unfused path: all 6 ops serialize on (0,0)-based grids.

        The fused path runs the RMS chain in parallel with the LN on
        disjoint cores.  Here everything serializes on the same grid.
        """
        torch.manual_seed(42)
        rows, cols = 2048, 512
        half = rows // 2

        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))})
        branch_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})

        def _shard_mem(cores, shard_h, shard_w):
            spec = ttnn.ShardSpec(cores, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

        stem_mem = _shard_mem(stem_cores, 256, 128)
        branch_mem = _shard_mem(branch_cores, 128, 256)

        stem_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 8),
            subblock_w=min(128 // 32, 4),
            block_h=256 // 32,
            block_w=128 // 32,
            inplace=False,
        )
        branch_ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),
            subblock_w=min(256 // 32, 4),
            block_h=128 // 32,
            block_w=256 // 32,
            inplace=False,
        )

        tt_input = ttnn.from_torch(
            torch.randn(1, 1, rows, cols, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=stem_mem,
        )
        # RMS weight on branch_cores
        w_shard = ttnn.ShardSpec(branch_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
        tw = ttnn.from_torch(
            torch.ones(1, 1, 1, cols, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=w_mem,
        )

        def unfused():
            u_stem = ttnn.layer_norm(
                tt_input,
                epsilon=1e-5,
                compute_kernel_config=COMPUTE_CONFIG,
                program_config=stem_ln_cfg,
                memory_config=stem_mem,
            )
            u_left = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols], memory_config=branch_mem)
            u_right = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols], memory_config=branch_mem)
            # 2x RMS on left half
            u_left = ttnn.rms_norm(
                u_left,
                weight=tw,
                epsilon=1e-5,
                program_config=branch_ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=branch_mem,
            )
            u_left = ttnn.rms_norm(
                u_left,
                weight=tw,
                epsilon=1e-5,
                program_config=branch_ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=branch_mem,
            )
            # LN on right half
            ttnn.layer_norm(
                u_right,
                epsilon=1e-5,
                program_config=branch_ln_cfg,
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=branch_mem,
            )

        if perf_mode == "device_fw":
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Asymmetric Branches Unfused: device_fw run")
        elif perf_mode == "cold_start":
            cold = _time_cold(unfused, device)
            print(f"\n  Asymmetric Branches Unfused: cold={cold:.2f}ms")
        elif perf_mode == "e2e":
            e2e = _time_e2e(unfused, device)
            print(f"\n  Asymmetric Branches Unfused: e2e={e2e:.3f}ms")


# =============================================================================
# Functional Demos (fused only, no unfused comparison)
# =============================================================================


# -----------------------------------------------------------------
# GlobalCircularBuffer mid-kernel write (fused only)
# -----------------------------------------------------------------


def test_global_circular_buffer_fused(device):
    from models.experimental.ops.descriptors.fusion import Sequential, Parallel

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

    # RECEIVER_DRAM_WRITER_SOURCE uses cb_in (not cb_out) — kept local to this test
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

    cold = _time_cold_fused(lambda: Parallel(Sequential(oa, ob), con).build(), device)

    result_recv = ttnn.to_torch(tt_output_recv)
    result_b = ttnn.to_torch(tt_output_b)

    passing_recv, pcc_recv = comp_pcc(torch_input_a, result_recv, pcc=0.999)
    passing_b, pcc_b = comp_pcc(torch_input_b, result_b, pcc=0.999)

    print(f"\n  GlobalCB Fused: cold={cold:.2f}ms  PCC: recv={pcc_recv:.4f} phase1={pcc_b:.4f}")
    assert passing_recv, f"Receiver PCC: {pcc_recv}"
    assert passing_b, f"Phase 1 PCC: {pcc_b}"


# -----------------------------------------------------------------
# Non-contiguous core grid ("swiss cheese")
# Data flows through a stem on scattered cores into two branches.
# Exercises unicast barrier release across gaps in the core grid.
#
#       col0  col1  col2  col3  col4  col5
# row0   X     X     X     X     X     X
# row1   X     X     X     X     X     X    branch A (18 cores)
# row2   .     .     .     .     .     .      <- gap
# row3   X     X     X     X     X     X
# row4   .     .     .     .     .     .      <- gap
# row5   X     X     X     X     X     X    <- branch B (6 cores)
#
# Stem = all 24 cores (3 CoreRanges — swiss cheese)
# Branch A = rows 0-1, 3 (2 CoreRanges — also swiss cheese)
# Branch B = row 5 (contiguous)
# -----------------------------------------------------------------


def _non_contiguous_grid_setup(device, num_tiles=4):
    torch.manual_seed(42)
    shape = [1, 1, 32, 32 * num_tiles]

    # Non-contiguous stem: rows 0-1, 3, 5 — gaps at rows 2 and 4
    all_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(5, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(5, 5)),
        }
    )
    # Branch A: rows 0-1, 3 (non-contiguous — also swiss cheese)
    cores_a = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(5, 3)),
        }
    )
    # Branch B: row 5 (contiguous)
    cores_b = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(5, 5)),
        }
    )

    dram = ttnn.DRAM_MEMORY_CONFIG
    t_in = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    t_mid = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    t_out_a = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    t_out_b = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )

    stem = _build_identity_op(t_in, t_mid, all_cores, num_tiles)
    op_a = _build_identity_op(t_mid, t_out_a, cores_a, num_tiles)
    op_b = _build_identity_op(t_mid, t_out_b, cores_b, num_tiles)

    return stem, op_a, op_b, t_in, t_out_a, t_out_b


@pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
def test_non_contiguous_core_grid_fused(device, perf_mode):
    from models.experimental.ops.descriptors.fusion import Sequential, Parallel

    stem, op_a, op_b, t_in, t_out_a, t_out_b = _non_contiguous_grid_setup(device)

    def build():
        return Sequential(stem, Parallel(op_a, op_b)).build()

    if perf_mode == "device_fw":
        fused = build()
        fused.launch()
        ttnn.synchronize_device(device)
        print("\n  Non-Contiguous Grid Fused: device_fw run")
    elif perf_mode == "cold_start":
        cold = _time_cold_fused(build, device)

        ref = ttnn.to_torch(t_in)
        p_a, pcc_a = comp_pcc(ref, ttnn.to_torch(t_out_a), pcc=0.999)
        p_b, pcc_b = comp_pcc(ref, ttnn.to_torch(t_out_b), pcc=0.999)

        print(f"\n  Non-Contiguous Grid Fused: cold={cold:.2f}ms PCC: A={pcc_a:.4f} B={pcc_b:.4f}")
        assert p_a, f"Branch A PCC: {pcc_a}"
        assert p_b, f"Branch B PCC: {pcc_b}"
    elif perf_mode == "e2e":
        fused = build()
        e2e = _time_e2e(fused.launch, device)

        ref = ttnn.to_torch(t_in)
        p_a, pcc_a = comp_pcc(ref, ttnn.to_torch(t_out_a), pcc=0.999)
        p_b, pcc_b = comp_pcc(ref, ttnn.to_torch(t_out_b), pcc=0.999)

        print(f"\n  Non-Contiguous Grid Fused: e2e={e2e:.3f}ms PCC: A={pcc_a:.4f} B={pcc_b:.4f}")
        assert p_a, f"Branch A PCC: {pcc_a}"
        assert p_b, f"Branch B PCC: {pcc_b}"


# -----------------------------------------------------------------
# Barrier Overhead Benchmark
#
# Chain N no-op phases via Sequential.  Each phase has 3 empty
# kernel_main() functions (reader, compute, writer) — zero kernel
# work, zero CBs, zero data.  The ONLY thing measured is the
# barrier synchronization mechanism itself.
#
# Core grid configs:
#   1  core  -> (0,0) only — local::sync only (no group::sync)
#   8  cores -> 1x8 row   — local + group (1 segment, 8 cores)
#   16 cores -> 2x8 rect  — local + group (1 segment, 16 cores)
#   64 cores -> 8x8 rect  — local + group (1 segment, 64 cores)
# -----------------------------------------------------------------


def _core_ranges_for(num_cores):
    grid = {1: (0, 0), 8: (7, 0), 16: (7, 1), 64: (7, 7)}
    if num_cores not in grid:
        raise ValueError(f"Unsupported num_cores={num_cores}")
    ex, ey = grid[num_cores]
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def _barrier_bench_setup(device, num_phases, num_cores):
    """Create N no-op ops for pure barrier benchmarking."""
    core_ranges = _core_ranges_for(num_cores)
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return [_build_noop_op(core_ranges, dummy) for _ in range(num_phases)]


@pytest.mark.parametrize("perf_mode", ["cold_start", "e2e", "device_fw"])
@pytest.mark.parametrize("num_cores", [1, 8, 16, 64])
@pytest.mark.parametrize("num_phases", [2, 3, 4, 5, 6])
def test_barrier_overhead(device, num_phases, num_cores, perf_mode):
    """Measure pure barrier mechanism cost by chaining no-op phases."""
    from models.experimental.ops.descriptors.fusion import Sequential

    ops = _barrier_bench_setup(device, num_phases, num_cores)

    def build_fused():
        return Sequential(*ops).build()

    if perf_mode == "device_fw":
        fused = build_fused()
        fused.launch()
        ttnn.synchronize_device(device)
        print(f"\n  Barrier bench: {num_phases} phases, {num_cores} cores (device_fw run)")
        return

    if perf_mode == "cold_start":
        cold = _time_cold_fused(build_fused, device)
        print(f"\n  Barrier bench: {num_phases} phases, {num_cores} cores cold={cold:.2f}ms")
        return

    # perf_mode == "e2e"
    # -- Fused timing --
    fused = build_fused()
    fused_e2e = _time_e2e(fused.launch, device)

    # -- Unfused timing: launch each phase as a separate 1-op fused kernel --
    unfused_ops = [Sequential(op).build() for op in ops]
    for uf in unfused_ops:
        uf.launch()
    ttnn.synchronize_device(device)

    def launch_unfused():
        for uf in unfused_ops:
            uf.launch()

    unfused_e2e = _time_e2e(launch_unfused, device)

    # -- 1-phase baseline for per-barrier calculation --
    ops_1 = _barrier_bench_setup(device, 1, num_cores)
    fused_1 = Sequential(*ops_1).build()
    baseline_e2e = _time_e2e(fused_1.launch, device)

    # Convert to microseconds
    fused_us = fused_e2e * 1000
    unfused_us = unfused_e2e * 1000
    baseline_us = baseline_e2e * 1000
    per_barrier_us = (fused_us - baseline_us) / (num_phases - 1)

    print(
        f"\n  Barrier Overhead ({num_cores} cores, {num_phases} phases): "
        f"fused={fused_us:.1f}us  unfused={unfused_us:.1f}us  "
        f"baseline(1-phase)={baseline_us:.1f}us  per_barrier={per_barrier_us:.1f}us"
    )
