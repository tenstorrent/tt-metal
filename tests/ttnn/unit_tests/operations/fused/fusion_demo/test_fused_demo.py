# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion Infrastructure Demo Suite

Five demos showcasing different fusion capabilities:

1. Basic 3-op chain (RMS -> Matmul -> RMS) with DRAM I/O on a 4x2 grid
2. Sharded 2-op chain (RMS -> LN) demonstrating pinned buffer address reassignment
3. All-matmul symmetric branching (7 matmuls) on full 8x8 grid — same core grids both paths
4. Two parallel independent chains on disjoint cores
5. GlobalCircularBuffer mid-kernel write to an external consumer

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


def _tt(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _cores(x1, y1, x2, y2):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


def _mm_config(grid_x, grid_y, in0_block_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=min(per_core_N, 4),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _compute(fp32=False, math_approx_mode=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32,
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
# GlobalCB Kernel Sources (for Demo 5)
# =============================================================================

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


# =============================================================================
# GlobalCB OpDescriptor Builders (for Demo 5)
# =============================================================================


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
    local_fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.DataType.BFLOAT16, page_size=TILE_SIZE_BF16)
    remote_fmt = ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.DataType.BFLOAT16, page_size=TILE_SIZE_BF16)
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


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestFusedDemo:
    """Fusion infrastructure demo tests.

    Each demo is split into fused and unfused tests with cold + warm timing.
    Cold = all caches cleared, warm = all caches populated.
    """

    # -----------------------------------------------------------------
    # Demo 1: RMS -> Matmul -> RMS (DRAM, 4x2 grid)
    # Parametrized: H=128 (small) vs H=1024 (large)
    # Input [256, H], matmul [H, H], 4x2 = 8 cores
    # -----------------------------------------------------------------

    def _demo1_setup(self, device, hidden):
        torch.manual_seed(42)
        M_tiles = 256 // 32  # 8
        N_tiles = hidden // 32
        core_range = _cores(0, 0, 3, 1)  # 4x2 = 8 cores
        mm_cfg = _mm_config(grid_x=4, grid_y=2, in0_block_w=4, per_core_M=M_tiles // 8, per_core_N=N_tiles)

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        return core_range, mm_cfg, torch_input, torch_w, torch_b

    @pytest.mark.parametrize("hidden", [128, 1024], ids=["small", "large"])
    def test_demo1_fused(self, device, hidden):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        core_range, mm_cfg, torch_input, torch_w, torch_b = self._demo1_setup(device, hidden)

        tt_in = _tt(torch_input, device)
        tt_w = _tt(torch_w, device)
        tt_b = _tt(torch_b, device)
        r1 = rms_norm.rms_norm(tt_in, core_range_set=core_range, weight=tt_w, epsilon=1e-5)
        m = matmul_desc(
            r1.output_tensors[0],
            tt_b,
            core_range_set=core_range,
            program_config=mm_cfg,
            compute_kernel_config=_compute(),
        )
        r2 = rms_norm.rms_norm(
            m.output_tensors[0], core_range_set=core_range, weight=_tt(torch_w, device), epsilon=1e-5
        )

        _, cold, warm = _time_fused(lambda: Sequential(r1, m, r2).build(device), device)
        fused_result = ttnn.to_torch(r2.output_tensors[0])

        # Unfused reference for PCC
        tt_in, tt_w, tt_B = _tt(torch_input, device), _tt(torch_w, device), _tt(torch_b, device)
        u1 = ttnn.rms_norm(tt_in, weight=tt_w, epsilon=1e-5)
        u2 = ttnn.matmul(u1, tt_B, program_config=mm_cfg, compute_kernel_config=_compute())
        ref = ttnn.to_torch(ttnn.rms_norm(u2, weight=tt_w, epsilon=1e-5))

        passing, pcc = comp_pcc(ref, fused_result, pcc=0.97)
        print(f"\n  Demo 1 Fused (H={hidden}): cold={cold:.2f}ms  warm={warm:.2f}ms  PCC={pcc:.6f}")
        assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("hidden", [128, 1024], ids=["small", "large"])
    def test_demo1_unfused(self, device, hidden):
        core_range, mm_cfg, torch_input, torch_w, torch_b = self._demo1_setup(device, hidden)
        tt_in, tt_w, tt_B = _tt(torch_input, device), _tt(torch_w, device), _tt(torch_b, device)

        def unfused():
            u1 = ttnn.rms_norm(tt_in, weight=tt_w, epsilon=1e-5)
            u2 = ttnn.matmul(u1, tt_B, program_config=mm_cfg, compute_kernel_config=_compute())
            return ttnn.rms_norm(u2, weight=tt_w, epsilon=1e-5)

        cold, warm = _time_cold_warm(unfused, device)
        print(f"\n  Demo 1 Unfused (H={hidden}): cold={cold:.2f}ms  warm={warm:.2f}ms")

    # -----------------------------------------------------------------
    # Demo 2: RMS -> LN (block-sharded, 4x4 grid)
    # Parametrized: rows=128/shard_h=32 (small) vs rows=512/shard_h=128 (large)
    # Cols fixed at 512, 4x4 = 16 cores
    # -----------------------------------------------------------------

    def _demo2_setup(self, device, rows):
        torch.manual_seed(42)
        cols = 512
        shard_h = rows // 4  # rows distributed across 4 core-rows
        shard_w = cols // 4  # 128 always
        block_h = shard_h // 32  # tile rows per core
        block_w = shard_w // 32  # tile cols per core = 4 always

        cores = _cores(0, 0, 3, 3)
        shard_spec = ttnn.ShardSpec(cores, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        compute_cfg = _compute(fp32=False)
        program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=4,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )

        torch_input = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, cols, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem,
        )
        tt_w = _tt(torch_w, device)

        return cores, sharded_mem, compute_cfg, program_cfg, tt_input, tt_w

    @pytest.mark.parametrize("rows", [128, 512], ids=["small", "large"])
    def test_demo2_fused(self, device, rows):
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm

        cores, sharded_mem, compute_cfg, program_cfg, tt_input, tt_w = self._demo2_setup(device, rows)

        r = rms_norm.rms_norm(
            tt_input,
            core_range_set=cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
            memory_config=sharded_mem,
        )
        ln = layer_norm.layer_norm(
            r.output_tensors[0],
            core_range_set=cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
            memory_config=sharded_mem,
        )

        _, cold, warm = _time_fused(lambda: Sequential(r, ln).build(device), device)
        fused_result = ttnn.to_torch(ln.output_tensors[0])

        # Unfused reference for PCC
        u1 = ttnn.rms_norm(
            tt_input,
            weight=tt_w,
            epsilon=1e-5,
            program_config=program_cfg,
            compute_kernel_config=compute_cfg,
            memory_config=sharded_mem,
        )
        ref = ttnn.to_torch(
            ttnn.layer_norm(
                u1,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=compute_cfg,
                memory_config=sharded_mem,
            )
        )

        passing, pcc = comp_pcc(ref, fused_result, pcc=0.98)
        print(f"\n  Demo 2 Fused (rows={rows}): cold={cold:.2f}ms  warm={warm:.2f}ms  PCC={pcc:.6f}")
        assert passing, f"PCC: {pcc}"

    @pytest.mark.parametrize("rows", [128, 512], ids=["small", "large"])
    def test_demo2_unfused(self, device, rows):
        cores, sharded_mem, compute_cfg, program_cfg, tt_input, tt_w = self._demo2_setup(device, rows)

        def unfused():
            u1 = ttnn.rms_norm(
                tt_input,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=compute_cfg,
                memory_config=sharded_mem,
            )
            return ttnn.layer_norm(
                u1,
                weight=tt_w,
                epsilon=1e-5,
                program_config=program_cfg,
                compute_kernel_config=compute_cfg,
                memory_config=sharded_mem,
            )

        cold, warm = _time_cold_warm(unfused, device)
        print(f"\n  Demo 2 Unfused (rows={rows}): cold={cold:.2f}ms  warm={warm:.2f}ms")

    # -----------------------------------------------------------------
    # Demo 3: All-matmul symmetric branching on full 8x8 grid
    #
    #     stem (8x8=64c)
    #       ├── left (8x4=32c)
    #       │     ├── ll (8x2=16c)
    #       │     └── lr (8x2=16c)
    #       └── right (8x4=32c)
    #             ├── rl (8x2=16c)
    #             └── rr (8x2=16c)
    #
    # All 7 matmuls: [2048,H] @ [H,H] → [2048,H].
    # Parametrized on hidden to compare small (128) vs large (1024) compute.
    # Using matmul everywhere gives both paths identical core grid
    # control via MatmulMultiCoreReuseProgramConfig.
    # -----------------------------------------------------------------

    def _demo3_setup(self, device, hidden, half_grid=False):
        torch.manual_seed(42)

        N_tiles = hidden // 32

        if half_grid:
            # Half grid: stem 4x4=16, branch 4x2=8, leaf 4x1=4
            M_tiles = 512 // 32  # 16
            stem_cores = _cores(0, 0, 3, 3)  # 4x4 = 16 cores
            left_cores = _cores(0, 0, 3, 1)  # 4x2 = 8 cores
            right_cores = _cores(0, 2, 3, 3)  # 4x2 = 8 cores
            ll_cores = _cores(0, 0, 3, 0)  # 4x1 = 4 cores
            lr_cores = _cores(0, 1, 3, 1)  # 4x1 = 4 cores
            rl_cores = _cores(0, 2, 3, 2)  # 4x1 = 4 cores
            rr_cores = _cores(0, 3, 3, 3)  # 4x1 = 4 cores
            stem_cfg = _mm_config(grid_x=4, grid_y=4, in0_block_w=4, per_core_M=M_tiles // 16, per_core_N=N_tiles)
            branch_cfg = _mm_config(grid_x=4, grid_y=2, in0_block_w=4, per_core_M=M_tiles // 8, per_core_N=N_tiles)
            leaf_cfg = _mm_config(grid_x=4, grid_y=1, in0_block_w=4, per_core_M=M_tiles // 4, per_core_N=N_tiles)
            rows = 512
        else:
            # Full grid: stem 8x8=64, branch 8x4=32, leaf 8x2=16
            M_tiles = 2048 // 32  # 64
            stem_cores = _cores(0, 0, 7, 7)  # 8x8 = 64 cores
            left_cores = _cores(0, 0, 7, 3)  # 8x4 = 32 cores
            right_cores = _cores(0, 4, 7, 7)  # 8x4 = 32 cores
            ll_cores = _cores(0, 0, 7, 1)  # 8x2 = 16 cores
            lr_cores = _cores(0, 2, 7, 3)  # 8x2 = 16 cores
            rl_cores = _cores(0, 4, 7, 5)  # 8x2 = 16 cores
            rr_cores = _cores(0, 6, 7, 7)  # 8x2 = 16 cores
            stem_cfg = _mm_config(grid_x=8, grid_y=8, in0_block_w=4, per_core_M=M_tiles // 64, per_core_N=N_tiles)
            branch_cfg = _mm_config(grid_x=8, grid_y=4, in0_block_w=4, per_core_M=M_tiles // 32, per_core_N=N_tiles)
            leaf_cfg = _mm_config(grid_x=8, grid_y=2, in0_block_w=4, per_core_M=M_tiles // 16, per_core_N=N_tiles)
            rows = 2048

        mm_compute = _compute()
        torch_A = torch.randn(1, 1, rows, hidden, dtype=torch.bfloat16)
        tt_A = _tt(torch_A, device)

        def _make_b():
            return ttnn.from_torch(
                torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tt_Bs = [_make_b()] * 7

        return (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            mm_compute,
            stem_cfg,
            branch_cfg,
            leaf_cfg,
            tt_A,
            tt_Bs,
        )

    @pytest.mark.parametrize(
        "hidden,half_grid", [(1024, False), (1024, True), (512, True)], ids=["full", "half", "half512"]
    )
    def test_demo3_fused(self, device, hidden, half_grid):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            mm_compute,
            stem_cfg,
            branch_cfg,
            leaf_cfg,
            tt_A,
            tt_Bs,
        ) = self._demo3_setup(device, hidden, half_grid=half_grid)

        m_stem = matmul_desc(
            tt_A, tt_Bs[0], core_range_set=stem_cores, program_config=stem_cfg, compute_kernel_config=mm_compute
        )
        m_left = matmul_desc(
            m_stem.output_tensors[0],
            tt_Bs[1],
            core_range_set=left_cores,
            program_config=branch_cfg,
            compute_kernel_config=mm_compute,
        )
        m_right = matmul_desc(
            m_stem.output_tensors[0],
            tt_Bs[2],
            core_range_set=right_cores,
            program_config=branch_cfg,
            compute_kernel_config=mm_compute,
        )
        m_ll = matmul_desc(
            m_left.output_tensors[0],
            tt_Bs[3],
            core_range_set=ll_cores,
            program_config=leaf_cfg,
            compute_kernel_config=mm_compute,
        )
        m_lr = matmul_desc(
            m_left.output_tensors[0],
            tt_Bs[4],
            core_range_set=lr_cores,
            program_config=leaf_cfg,
            compute_kernel_config=mm_compute,
        )
        m_rl = matmul_desc(
            m_right.output_tensors[0],
            tt_Bs[5],
            core_range_set=rl_cores,
            program_config=leaf_cfg,
            compute_kernel_config=mm_compute,
        )
        m_rr = matmul_desc(
            m_right.output_tensors[0],
            tt_Bs[6],
            core_range_set=rr_cores,
            program_config=leaf_cfg,
            compute_kernel_config=mm_compute,
        )

        _, cold, warm = _time_fused(
            lambda: Sequential(
                m_stem,
                Parallel(
                    Sequential(m_left, Parallel(m_ll, m_lr)),
                    Sequential(m_right, Parallel(m_rl, m_rr)),
                ),
            ).build(device),
            device,
        )
        result_ll = ttnn.to_torch(m_ll.output_tensors[0])
        result_lr = ttnn.to_torch(m_lr.output_tensors[0])
        result_rl = ttnn.to_torch(m_rl.output_tensors[0])
        result_rr = ttnn.to_torch(m_rr.output_tensors[0])

        # Unfused reference for PCC
        u_stem = ttnn.matmul(tt_A, tt_Bs[0], program_config=stem_cfg, compute_kernel_config=mm_compute)
        u_left = ttnn.matmul(u_stem, tt_Bs[1], program_config=branch_cfg, compute_kernel_config=mm_compute)
        u_right = ttnn.matmul(u_stem, tt_Bs[2], program_config=branch_cfg, compute_kernel_config=mm_compute)
        u_ll = ttnn.matmul(u_left, tt_Bs[3], program_config=leaf_cfg, compute_kernel_config=mm_compute)
        u_lr = ttnn.matmul(u_left, tt_Bs[4], program_config=leaf_cfg, compute_kernel_config=mm_compute)
        u_rl = ttnn.matmul(u_right, tt_Bs[5], program_config=leaf_cfg, compute_kernel_config=mm_compute)
        u_rr = ttnn.matmul(u_right, tt_Bs[6], program_config=leaf_cfg, compute_kernel_config=mm_compute)

        p_ll, pcc_ll = comp_pcc(ttnn.to_torch(u_ll), result_ll, pcc=0.97)
        p_lr, pcc_lr = comp_pcc(ttnn.to_torch(u_lr), result_lr, pcc=0.97)
        p_rl, pcc_rl = comp_pcc(ttnn.to_torch(u_rl), result_rl, pcc=0.97)
        p_rr, pcc_rr = comp_pcc(ttnn.to_torch(u_rr), result_rr, pcc=0.97)

        grid = "half" if half_grid else "full"
        print(
            f"\n  Demo 3 Fused ({grid}, H={hidden}): cold={cold:.2f}ms  warm={warm:.2f}ms  PCC: ll={pcc_ll:.4f} lr={pcc_lr:.4f} rl={pcc_rl:.4f} rr={pcc_rr:.4f}"
        )
        assert p_ll, f"ll PCC: {pcc_ll}"
        assert p_lr, f"lr PCC: {pcc_lr}"
        assert p_rl, f"rl PCC: {pcc_rl}"
        assert p_rr, f"rr PCC: {pcc_rr}"

    @pytest.mark.parametrize(
        "hidden,half_grid", [(1024, False), (1024, True), (512, True)], ids=["full", "half", "half512"]
    )
    def test_demo3_unfused(self, device, hidden, half_grid):
        (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            mm_compute,
            stem_cfg,
            branch_cfg,
            leaf_cfg,
            tt_A,
            tt_Bs,
        ) = self._demo3_setup(device, hidden, half_grid=half_grid)

        def unfused():
            u_stem = ttnn.matmul(tt_A, tt_Bs[0], program_config=stem_cfg, compute_kernel_config=mm_compute)
            u_left = ttnn.matmul(u_stem, tt_Bs[1], program_config=branch_cfg, compute_kernel_config=mm_compute)
            ttnn.matmul(u_left, tt_Bs[3], program_config=leaf_cfg, compute_kernel_config=mm_compute)
            ttnn.matmul(u_left, tt_Bs[4], program_config=leaf_cfg, compute_kernel_config=mm_compute)
            u_right = ttnn.matmul(u_stem, tt_Bs[2], program_config=branch_cfg, compute_kernel_config=mm_compute)
            ttnn.matmul(u_right, tt_Bs[5], program_config=leaf_cfg, compute_kernel_config=mm_compute)
            ttnn.matmul(u_right, tt_Bs[6], program_config=leaf_cfg, compute_kernel_config=mm_compute)

        cold, warm = _time_cold_warm(unfused, device)
        grid = "half" if half_grid else "full"
        print(f"\n  Demo 3 Unfused ({grid}, H={hidden}): cold={cold:.2f}ms  warm={warm:.2f}ms")

    # -----------------------------------------------------------------
    # Demo 4: Two parallel 2-op chains
    # -----------------------------------------------------------------

    def _demo4_setup(self, device):
        torch.manual_seed(42)
        cores_a = _cores(0, 0, 3, 3)
        cores_b = _cores(4, 0, 7, 3)
        hidden = 128
        mm_cfg = _mm_config(grid_x=4, grid_y=4, in0_block_w=4, per_core_M=1, per_core_N=4)

        torch_a = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_B = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        ta = _tt(torch_a, device)
        tb = _tt(torch_b, device)
        tw = _tt(torch_w, device)
        tbi = _tt(torch_bias, device)
        tB = _tt(torch_B, device)

        return cores_a, cores_b, mm_cfg, ta, tb, tw, tbi, tB

    def test_demo4_fused(self, device):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        cores_a, cores_b, mm_cfg, ta, tb, tw, tbi, tB = self._demo4_setup(device)

        la = layer_norm.layer_norm(
            ta,
            core_range_set=cores_a,
            weight=tw,
            bias=tbi,
            epsilon=1e-5,
            compute_kernel_config=_compute(),
        )
        ma = matmul_desc(
            la.output_tensors[0],
            tB,
            core_range_set=cores_a,
            program_config=mm_cfg,
            compute_kernel_config=_compute(),
        )
        rb = rms_norm.rms_norm(tb, core_range_set=cores_b, weight=tw, epsilon=1e-5)
        mb = matmul_desc(
            rb.output_tensors[0],
            tB,
            core_range_set=cores_b,
            program_config=mm_cfg,
            compute_kernel_config=_compute(),
        )

        fused = [None]

        def build_and_launch():
            fused[0] = Parallel(Sequential(la, ma), Sequential(rb, mb)).build(device)
            fused[0].launch()

        cold, warm = _time_cold_warm(build_and_launch, device, warm_fn=lambda: fused[0].launch())
        result_a = ttnn.to_torch(ma.output_tensors[0])
        result_b = ttnn.to_torch(mb.output_tensors[0])

        # Quick unfused reference for PCC
        ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=_compute())
        ua2 = ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=_compute())
        ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5)
        ub2 = ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=_compute())

        p_a, pcc_a = comp_pcc(ttnn.to_torch(ua2), result_a, pcc=0.97)
        p_b, pcc_b = comp_pcc(ttnn.to_torch(ub2), result_b, pcc=0.97)

        print(f"\n  Demo 4 Fused: cold={cold:.2f}ms  warm={warm:.2f}ms  PCC: a={pcc_a:.4f} b={pcc_b:.4f}")
        assert p_a, f"Chain A PCC: {pcc_a}"
        assert p_b, f"Chain B PCC: {pcc_b}"

    def test_demo4_unfused(self, device):
        cores_a, cores_b, mm_cfg, ta, tb, tw, tbi, tB = self._demo4_setup(device)

        def unfused():
            ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=_compute())
            ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=_compute())
            ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5)
            ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=_compute())

        cold, warm = _time_cold_warm(unfused, device)
        print(f"\n  Demo 4 Unfused: cold={cold:.2f}ms  warm={warm:.2f}ms")

    # -----------------------------------------------------------------
    # Demo 5: GlobalCircularBuffer mid-kernel write (fused only)
    # -----------------------------------------------------------------

    def test_demo5_fused(self, device):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        torch.manual_seed(42)
        num_tiles = 8
        shape = [1, 1, 32, 32 * num_tiles]

        torch_input_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_input_b = torch.randn(shape, dtype=torch.bfloat16)

        sender_core = ttnn.CoreCoord(0, 0)
        sender_range = _cores(0, 0, 0, 0)
        receiver_range = _cores(1, 0, 1, 0)

        gcb_size = TILE_SIZE_BF16 * 2
        gcb = ttnn.create_global_circular_buffer(device, [(sender_core, receiver_range)], gcb_size)

        tia = _tt(torch_input_a, device)
        tib = _tt(torch_input_b, device)
        tt_output_b = _tt(torch.zeros(shape, dtype=torch.bfloat16), device)
        tt_output_recv = _tt(torch.zeros(shape, dtype=torch.bfloat16), device)
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

    # -----------------------------------------------------------------
    # Demo 6: All-layernorm symmetric branching on 2x8 grid
    #
    #     stem (2x8=16c) — LN on [512,1024]
    #       ├── left (1x8=8c)
    #       │     ├── ll (1x4=4c)
    #       │     └── lr (1x4=4c)
    #       └── right (1x8=8c)
    #             ├── rl (1x4=4c)
    #             └── rr (1x4=4c)
    #
    # DRAM interleaved, no weight/bias. Each phase re-normalizes the
    # same [512,1024] tensor but on a shrinking core subset.
    # -----------------------------------------------------------------

    def _demo6_setup(self, device):
        torch.manual_seed(42)
        rows, cols = 512, 1024

        stem_cores = _cores(0, 0, 1, 7)  # 2x8 = 16 cores
        left_cores = _cores(0, 0, 0, 7)  # 1x8 = 8 cores
        right_cores = _cores(1, 0, 1, 7)  # 1x8 = 8 cores
        ll_cores = _cores(0, 0, 0, 3)  # 1x4 = 4 cores
        lr_cores = _cores(0, 4, 0, 7)  # 1x4 = 4 cores
        rl_cores = _cores(1, 0, 1, 3)  # 1x4 = 4 cores
        rr_cores = _cores(1, 4, 1, 7)  # 1x4 = 4 cores

        compute_cfg = _compute(fp32=False)
        torch_input = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
        tt_input = _tt(torch_input, device)

        return (stem_cores, left_cores, right_cores, ll_cores, lr_cores, rl_cores, rr_cores, compute_cfg, tt_input)

    def test_demo6_fused(self, device):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice_op

        (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            compute_cfg,
            tt_input,
        ) = self._demo6_setup(device)

        rows, cols = 512, 1024
        half = rows // 2  # 256
        quarter = rows // 4  # 128

        # Level 0: stem LN on 16 cores, full [1,1,512,1024]
        ln_stem = layer_norm.layer_norm(
            tt_input, core_range_set=stem_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )

        # Level 1: slice top/bottom halves, then LN on 8 cores each
        sl_top = slice_op(ln_stem.output_tensors[0], [0, 0, 0, 0], [1, 1, half, cols], core_range_set=left_cores)
        sl_bot = slice_op(ln_stem.output_tensors[0], [0, 0, half, 0], [1, 1, rows, cols], core_range_set=right_cores)

        ln_left = layer_norm.layer_norm(
            sl_top.output_tensors[0], core_range_set=left_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_right = layer_norm.layer_norm(
            sl_bot.output_tensors[0], core_range_set=right_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )

        # Level 2: slice each half into quarters, then LN on 4 cores each
        sl_tl = slice_op(ln_left.output_tensors[0], [0, 0, 0, 0], [1, 1, quarter, cols], core_range_set=ll_cores)
        sl_bl = slice_op(ln_left.output_tensors[0], [0, 0, quarter, 0], [1, 1, half, cols], core_range_set=lr_cores)
        sl_tr = slice_op(ln_right.output_tensors[0], [0, 0, 0, 0], [1, 1, quarter, cols], core_range_set=rl_cores)
        sl_br = slice_op(ln_right.output_tensors[0], [0, 0, quarter, 0], [1, 1, half, cols], core_range_set=rr_cores)

        ln_ll = layer_norm.layer_norm(
            sl_tl.output_tensors[0], core_range_set=ll_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_lr = layer_norm.layer_norm(
            sl_bl.output_tensors[0], core_range_set=lr_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_rl = layer_norm.layer_norm(
            sl_tr.output_tensors[0], core_range_set=rl_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_rr = layer_norm.layer_norm(
            sl_br.output_tensors[0], core_range_set=rr_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )

        _, cold, warm = _time_fused(
            lambda: Sequential(
                ln_stem,
                Parallel(
                    Sequential(sl_top, ln_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
                    Sequential(sl_bot, ln_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
                ),
            ).build(device),
            device,
        )

        # Unfused reference for PCC (just stem → slice top → left → slice tl → ll path)
        u_stem = ttnn.layer_norm(tt_input, epsilon=1e-5, compute_kernel_config=compute_cfg)
        u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols])
        u_left = ttnn.layer_norm(u_top, epsilon=1e-5, compute_kernel_config=compute_cfg)
        u_tl = ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, cols])
        ref_ll = ttnn.to_torch(ttnn.layer_norm(u_tl, epsilon=1e-5, compute_kernel_config=compute_cfg))
        result_ll = ttnn.to_torch(ln_ll.output_tensors[0])

        passing, pcc = comp_pcc(ref_ll, result_ll, pcc=0.97)
        print(f"\n  Demo 6 Fused: cold={cold:.2f}ms  warm={warm:.2f}ms  PCC(ll)={pcc:.6f}")
        assert passing, f"PCC: {pcc}"

    def test_demo6_unfused(self, device):
        (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            compute_cfg,
            tt_input,
        ) = self._demo6_setup(device)

        rows, cols = 512, 1024
        half = rows // 2
        quarter = rows // 4

        def unfused():
            u_stem = ttnn.layer_norm(tt_input, epsilon=1e-5, compute_kernel_config=compute_cfg)
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols])
            u_bot = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols])
            u_left = ttnn.layer_norm(u_top, epsilon=1e-5, compute_kernel_config=compute_cfg)
            u_right = ttnn.layer_norm(u_bot, epsilon=1e-5, compute_kernel_config=compute_cfg)
            ttnn.layer_norm(
                ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, cols]), epsilon=1e-5, compute_kernel_config=compute_cfg
            )
            ttnn.layer_norm(
                ttnn.slice(u_left, [0, 0, quarter, 0], [1, 1, half, cols]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )
            ttnn.layer_norm(
                ttnn.slice(u_right, [0, 0, 0, 0], [1, 1, quarter, cols]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )
            ttnn.layer_norm(
                ttnn.slice(u_right, [0, 0, quarter, 0], [1, 1, half, cols]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )

        cold, warm = _time_cold_warm(unfused, device)
        print(f"\n  Demo 6 Unfused: cold={cold:.2f}ms  warm={warm:.2f}ms")

    # =================================================================
    # Demo 7: LN → Slice → Matmul → Slice → LN  (heterogeneous tree)
    # =================================================================
    #
    # Same balanced binary tree topology as demo 6, but the two "middle"
    # nodes are matmul ops that each read a different B weight tensor.
    #
    #   Level 0:  LN_stem            (16 cores)  [1,1,8192,1024]
    #   Level 1:  Slice → MM_left    ( 8 cores)  [1,1,4096,1024] × B_left → [1,1,4096,256]
    #             Slice → MM_right   ( 8 cores)  [1,1,4096,1024] × B_right → [1,1,4096,256]
    #   Level 2:  Slice → LN_ll/lr   ( 4 cores)  [1,1,2048,256]
    #             Slice → LN_rl/rr   ( 4 cores)  [1,1,2048,256]

    def _demo7_setup(self, device):
        torch.manual_seed(42)
        rows, cols = 8192, 1024
        mm_n = 256  # matmul output width (B = [1024, 256] = 512KB)

        stem_cores = _cores(0, 0, 1, 7)  # 2x8 = 16 cores
        left_cores = _cores(0, 0, 0, 7)  # 1x8 = 8 cores
        right_cores = _cores(1, 0, 1, 7)  # 1x8 = 8 cores
        ll_cores = _cores(0, 0, 0, 3)  # 1x4 = 4 cores
        lr_cores = _cores(0, 4, 0, 7)  # 1x4 = 4 cores
        rl_cores = _cores(1, 0, 1, 3)  # 1x4 = 4 cores
        rr_cores = _cores(1, 4, 1, 7)  # 1x4 = 4 cores

        compute_cfg = _compute(fp32=False)

        # Matmul config for 8 cores (1x8 grid):
        # A=[1,1,4096,1024] → M=128 tiles, K=32 tiles
        # B=[1,1,1024,256] → output [1,1,4096,256]
        mm_n_tiles = mm_n // 32  # 8
        mm_cfg = _mm_config(
            grid_x=1,
            grid_y=8,
            in0_block_w=4,
            per_core_M=16,  # 128 M-tiles / 8 cores = 16
            per_core_N=mm_n_tiles,  # 8 N-tiles / 1 = 8
        )

        torch_input = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
        tt_input = _tt(torch_input, device)

        # B in L1 interleaved: eliminates DRAM bank contention when parallel
        # matmuls read B simultaneously on disjoint core groups.
        def _tt_l1(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        tt_B_left = _tt_l1(torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16))
        tt_B_right = _tt_l1(torch.randn(1, 1, cols, mm_n, dtype=torch.bfloat16))

        return (
            stem_cores,
            left_cores,
            right_cores,
            ll_cores,
            lr_cores,
            rl_cores,
            rr_cores,
            compute_cfg,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
        )

    def _demo7_make_ops(self, device):
        """Create all OpDescriptors for Demo 7's tree."""
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
            compute_cfg,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
        ) = self._demo7_setup(device)

        rows, cols = 8192, 1024
        half = rows // 2
        quarter = rows // 4

        # Level 0: stem LN on 16 cores
        ln_stem = layer_norm.layer_norm(
            tt_input, core_range_set=stem_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )

        # Level 1: slice then matmul
        sl_top = slice_op(ln_stem.output_tensors[0], [0, 0, 0, 0], [1, 1, half, cols], core_range_set=left_cores)
        sl_bot = slice_op(ln_stem.output_tensors[0], [0, 0, half, 0], [1, 1, rows, cols], core_range_set=right_cores)

        mm_left = matmul_desc(
            sl_top.output_tensors[0],
            tt_B_left,
            core_range_set=left_cores,
            program_config=mm_cfg,
            compute_kernel_config=compute_cfg,
        )
        mm_right = matmul_desc(
            sl_bot.output_tensors[0],
            tt_B_right,
            core_range_set=right_cores,
            program_config=mm_cfg,
            compute_kernel_config=compute_cfg,
        )

        # Level 2: slice then LN (post-matmul width = mm_n)
        sl_tl = slice_op(mm_left.output_tensors[0], [0, 0, 0, 0], [1, 1, quarter, mm_n], core_range_set=ll_cores)
        sl_bl = slice_op(mm_left.output_tensors[0], [0, 0, quarter, 0], [1, 1, half, mm_n], core_range_set=lr_cores)
        sl_tr = slice_op(mm_right.output_tensors[0], [0, 0, 0, 0], [1, 1, quarter, mm_n], core_range_set=rl_cores)
        sl_br = slice_op(mm_right.output_tensors[0], [0, 0, quarter, 0], [1, 1, half, mm_n], core_range_set=rr_cores)

        ln_ll = layer_norm.layer_norm(
            sl_tl.output_tensors[0], core_range_set=ll_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_lr = layer_norm.layer_norm(
            sl_bl.output_tensors[0], core_range_set=lr_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_rl = layer_norm.layer_norm(
            sl_tr.output_tensors[0], core_range_set=rl_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
        )
        ln_rr = layer_norm.layer_norm(
            sl_br.output_tensors[0], core_range_set=rr_cores, epsilon=1e-5, compute_kernel_config=compute_cfg
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
            compute_cfg,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
        )

    def _demo7_build_fused(self, device, ops):
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr) = ops
        return Sequential(
            ln_stem,
            Parallel(
                Sequential(sl_top, mm_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
                Sequential(sl_bot, mm_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
            ),
        ).build(device)

    # Set to True for Tracy profiling: skips timing loops, runs once only.
    _SINGLE_RUN_ONLY = False

    def test_demo7_fused(self, device):
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
            compute_cfg,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
        ) = self._demo7_make_ops(device)

        rows, cols = 8192, 1024
        half = rows // 2
        quarter = rows // 4
        ops = (ln_stem, sl_top, sl_bot, mm_left, mm_right, sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr)

        if self._SINGLE_RUN_ONLY:
            fused = self._demo7_build_fused(device, ops)
            fused.launch()
            ttnn.synchronize_device(device)
            print("\n  Demo 7 Fused: single run (for Tracy)")
        else:
            # Cold start
            _, cold, _ = _time_fused(lambda: self._demo7_build_fused(device, ops), device)

            # Steady-state e2e
            fused = self._demo7_build_fused(device, ops)
            e2e = _time_steady_state(fused.launch, device)

            # Unfused reference for PCC
            u_stem = ttnn.layer_norm(tt_input, epsilon=1e-5, compute_kernel_config=compute_cfg)
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols])
            u_left = ttnn.matmul(u_top, tt_B_left, program_config=mm_cfg, compute_kernel_config=compute_cfg)
            u_tl = ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, mm_n])
            ref_ll = ttnn.to_torch(ttnn.layer_norm(u_tl, epsilon=1e-5, compute_kernel_config=compute_cfg))
            result_ll = ttnn.to_torch(ln_ll.output_tensors[0])

            passing, pcc = comp_pcc(ref_ll, result_ll, pcc=0.97)
            print(f"\n  Demo 7 Fused: cold={cold:.2f}ms  e2e={e2e:.3f}ms  PCC(ll)={pcc:.6f}")
            assert passing, f"PCC: {pcc}"

    def test_demo7_unfused(self, device):
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            compute_cfg,
            mm_cfg,
            mm_n,
            tt_input,
            tt_B_left,
            tt_B_right,
        ) = self._demo7_make_ops(device)

        rows, cols = 8192, 1024
        half = rows // 2
        quarter = rows // 4

        def unfused():
            u_stem = ttnn.layer_norm(tt_input, epsilon=1e-5, compute_kernel_config=compute_cfg)
            u_top = ttnn.slice(u_stem, [0, 0, 0, 0], [1, 1, half, cols])
            u_bot = ttnn.slice(u_stem, [0, 0, half, 0], [1, 1, rows, cols])
            u_left = ttnn.matmul(u_top, tt_B_left, program_config=mm_cfg, compute_kernel_config=compute_cfg)
            u_right = ttnn.matmul(u_bot, tt_B_right, program_config=mm_cfg, compute_kernel_config=compute_cfg)
            ttnn.layer_norm(
                ttnn.slice(u_left, [0, 0, 0, 0], [1, 1, quarter, mm_n]), epsilon=1e-5, compute_kernel_config=compute_cfg
            )
            ttnn.layer_norm(
                ttnn.slice(u_left, [0, 0, quarter, 0], [1, 1, half, mm_n]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )
            ttnn.layer_norm(
                ttnn.slice(u_right, [0, 0, 0, 0], [1, 1, quarter, mm_n]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )
            ttnn.layer_norm(
                ttnn.slice(u_right, [0, 0, quarter, 0], [1, 1, half, mm_n]),
                epsilon=1e-5,
                compute_kernel_config=compute_cfg,
            )

        if self._SINGLE_RUN_ONLY:
            unfused()
            ttnn.synchronize_device(device)
            print("\n  Demo 7 Unfused: single run (for Tracy)")
        else:
            # Cold start
            cold, _ = _time_cold_warm(unfused, device)

            # Steady-state e2e
            e2e = _time_steady_state(unfused, device)

            print(f"\n  Demo 7 Unfused: cold={cold:.2f}ms  e2e={e2e:.3f}ms")
