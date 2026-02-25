# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion Infrastructure Demo Suite

Five demos showcasing different fusion capabilities:

1. Basic 3-op chain (RMS -> Matmul -> RMS) with DRAM I/O on a 4x2 grid
2. Sharded 2-op chain (RMS -> LN) demonstrating pinned buffer address reassignment
3. Branching topology (5 ops) on full 8x8 grid (segment barrier measurement)
4. Two parallel independent chains on disjoint cores
5. GlobalCircularBuffer mid-kernel write to an external consumer
"""

import time

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


# =============================================================================
# Golden Reference Functions
# =============================================================================


def torch_layer_norm(x, weight, bias=None, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    out = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def torch_rms_norm(x, weight, eps=1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x / rms
    if weight is not None:
        out = out * weight
    return out


# =============================================================================
# Helpers
# =============================================================================


def _tt(t, device):
    """Move tensor to device with DRAM interleaved memory config."""
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _cores(x1, y1, x2, y2):
    """Create a CoreRangeSet from corner coordinates."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


def _mm_config(grid_x, grid_y, in0_block_w, per_core_M, per_core_N):
    """Create a MatmulMultiCoreReuseProgramConfig."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=min(per_core_N, 4),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _compute(fp32=False, math_approx_mode=True):
    """Create a WormholeComputeKernelConfig.

    Defaults match RMS norm's internal defaults (fp32=False, math_approx_mode=True).
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32,
    )


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
    """Phase 0: read from DRAM -> compute identity -> push to GlobalCB."""
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
    """Identity copy: DRAM -> compute -> DRAM."""
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
    """Consumer: wait on GlobalCB -> write to DRAM."""
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


@pytest.fixture(params=[1, 2], ids=["warmup", "timed"], scope="class")
def iteration(request):
    return request.param


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestFusedDemo:
    """Fusion infrastructure demo tests.

    The suite runs twice via the ``iteration`` parameter.  The first pass
    (warmup) populates the JIT cache for ALL kernels -- fused and unfused.
    The second pass (timed) is all cache hits and provides accurate timing.
    """

    # -----------------------------------------------------------------
    # Demo 1: RMS -> Matmul -> RMS (DRAM, 4x2 grid)
    # -----------------------------------------------------------------

    def test_demo1_rms_matmul_rms(self, device, iteration):
        """3-op chain: RMS -> Matmul -> RMS on a 4x2 grid with DRAM I/O.

        Demonstrates basic sequential chaining of heterogeneous ops.
        Intermediate results stay in L1 between phases -- no DRAM round-trips.

        Setup:
            Input:  (1, 1, 256, 128) -- 8x4 tiles, BF16, DRAM interleaved
            B:      (1, 1, 128, 128) -- 4x4 tiles
            Grid:   4x2 = 8 cores (8 M-tile rows, 1 per core)
            Compute: fp32=False, math_approx=True, HiFi4 (all phases)

        Comparison: ttnn.rms_norm -> ttnn.matmul -> ttnn.rms_norm
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        hidden = 128
        core_range = _cores(0, 0, 3, 1)
        mm_cfg = _mm_config(grid_x=4, grid_y=2, in0_block_w=4, per_core_M=1, per_core_N=4)

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        # ---- Build + launch fused ----
        factory_start = time.perf_counter()
        r1 = rms_norm.rms_norm(
            _tt(torch_input, device),
            core_range_set=core_range,
            weight=_tt(torch_w, device),
            epsilon=1e-5,
        )
        m = matmul_desc(
            r1.output_tensors[0],
            _tt(torch_b, device),
            core_range_set=core_range,
            program_config=mm_cfg,
            compute_kernel_config=_compute(),
        )
        r2 = rms_norm.rms_norm(
            m.output_tensors[0],
            core_range_set=core_range,
            weight=_tt(torch_w, device),
            epsilon=1e-5,
        )
        factory_time = time.perf_counter() - factory_start
        build_start = time.perf_counter()
        fused = Sequential(r1, m, r2).build(device)
        build_time = time.perf_counter() - build_start

        outputs = composite.launch([fused])
        fused_result = ttnn.to_torch(outputs[0][0])

        # ---- Unfused path ----
        tt_input = _tt(torch_input, device)
        tt_w = _tt(torch_w, device)
        tt_B = _tt(torch_b, device)

        # Warmup unfused ops (populate C++ program cache)
        _u0 = ttnn.rms_norm(tt_input, weight=tt_w, epsilon=1e-5)
        _u0 = ttnn.matmul(_u0, tt_B, program_config=mm_cfg, compute_kernel_config=_compute())
        _u0 = ttnn.rms_norm(_u0, weight=tt_w, epsilon=1e-5)
        del _u0

        # Timed unfused path (C++ program cache hit)
        t_u0 = time.perf_counter()
        u1 = ttnn.rms_norm(tt_input, weight=tt_w, epsilon=1e-5)
        t_u1 = time.perf_counter()
        u2 = ttnn.matmul(u1, tt_B, program_config=mm_cfg, compute_kernel_config=_compute())
        t_u2 = time.perf_counter()
        u3 = ttnn.rms_norm(u2, weight=tt_w, epsilon=1e-5)
        t_u3 = time.perf_counter()
        unfused_result = ttnn.to_torch(u3)

        # ---- Golden ----
        temp = torch_rms_norm(torch_input.float(), torch_w.float())
        temp = temp @ torch_b.float()
        golden = torch_rms_norm(temp, torch_w.float())

        passing_f, pcc_f = comp_pcc(golden, fused_result, pcc=0.97)
        passing_u, pcc_u = comp_pcc(golden, unfused_result, pcc=0.97)

        if iteration == 2:
            print(f"\n{'='*60}")
            print(f"Demo 1: RMS -> Matmul -> RMS (4x2 grid, DRAM)")
            print(f"  Factory (tensor alloc + descriptors): {factory_time*1000:.2f} ms")
            print(f"  Build (Sequential.build): {build_time*1000:.2f} ms")
            print(f"  Unfused host time per op (C++ cache hit):")
            print(f"    rms_norm: {1000*(t_u1-t_u0):.3f} ms")
            print(f"    matmul:   {1000*(t_u2-t_u1):.3f} ms")
            print(f"    rms_norm: {1000*(t_u3-t_u2):.3f} ms")
            print(f"    total:    {1000*(t_u3-t_u0):.3f} ms")
            print(f"  PCC: fused={pcc_f:.6f}  unfused={pcc_u:.6f}")
            print(f"{'='*60}")

        assert passing_f, f"Fused PCC: {pcc_f}"
        assert passing_u, f"Unfused PCC: {pcc_u}"

    # -----------------------------------------------------------------
    # Demo 2: RMS -> LN (block-sharded, 4x4 grid)
    # -----------------------------------------------------------------

    def test_demo2_rms_ln_sharded(self, device, iteration):
        """2-op chain: RMS -> LN with block-sharded input/output on 4x4 grid.

        Demonstrates pinned buffer address reassignment in the CB allocator.
        Sharded CBs have their L1 buffer addresses pinned to the shard location.
        The fusion CB allocator detects this and preserves the pinning while
        still pool-allocating other CB slots for sharing between phases.

        Setup:
            Input:  (1, 1, 128, 512) -- 4x16 tiles, BF16, block-sharded
            Grid:   4x4 = 16 cores, shard (32, 128) = 1x4 tiles per core
            Compute: fp32=False, math_approx=True, HiFi4 (all phases)

        Comparison: ttnn.rms_norm -> ttnn.layer_norm (sharded program config)
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        cores = _cores(0, 0, 3, 3)
        shard_spec = ttnn.ShardSpec(cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        compute_cfg = _compute(fp32=False)
        program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=4,
            block_h=1,
            block_w=4,
            inplace=False,
        )

        torch_input = torch.randn(1, 1, 128, 512, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, 512, dtype=torch.bfloat16)

        # ---- Build + launch fused ----
        build_start = time.perf_counter()
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem,
        )
        tt_w = _tt(torch_w, device)
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
        fused = Sequential(r, ln).build(device)
        build_time = time.perf_counter() - build_start

        outputs = composite.launch([fused])
        fused_result = ttnn.to_torch(outputs[0][0])

        # ---- Unfused path ----
        u1 = ttnn.rms_norm(
            tt_input,
            weight=tt_w,
            epsilon=1e-5,
            program_config=program_cfg,
            compute_kernel_config=compute_cfg,
            memory_config=sharded_mem,
        )
        u2 = ttnn.layer_norm(
            u1,
            weight=tt_w,
            epsilon=1e-5,
            program_config=program_cfg,
            compute_kernel_config=compute_cfg,
            memory_config=sharded_mem,
        )
        unfused_result = ttnn.to_torch(u2)

        # ---- Golden ----
        temp = torch_rms_norm(torch_input.float(), torch_w.float())
        golden = torch_layer_norm(temp, torch_w.float())

        passing_f, pcc_f = comp_pcc(golden, fused_result, pcc=0.98)
        passing_u, pcc_u = comp_pcc(golden, unfused_result, pcc=0.98)

        if iteration == 2:
            print(f"\n{'='*60}")
            print(f"Demo 2: RMS -> LN (4x4 grid, block-sharded)")
            print(f"  Fused compile: {build_time*1000:.2f} ms")
            print(f"  PCC: fused={pcc_f:.6f}  unfused={pcc_u:.6f}")
            print(f"{'='*60}")

        assert passing_f, f"Fused PCC: {pcc_f}"
        assert passing_u, f"Unfused PCC: {pcc_u}"

    # -----------------------------------------------------------------
    # Demo 4: Two parallel 2-op chains
    # -----------------------------------------------------------------

    def test_demo4_parallel_chains(self, device, iteration):
        """Two independent 2-op chains on disjoint 4x4 cores, fused in parallel.

        Chain A: LN -> Matmul on (0,0)-(3,3) = 16 cores
        Chain B: RMS -> Matmul on (4,0)-(7,3) = 16 cores

        Both chains execute in a single kernel dispatch. The hardware runs
        them simultaneously on separate cores with no inter-chain synchronization.

        Setup:
            Input A: (1, 1, 512, 128) -- 16x4 tiles
            Input B: (1, 1, 512, 128) -- 16x4 tiles
            B weight: (1, 1, 128, 128) -- 4x4 tiles
            Grid A: (0,0)-(3,3) = 4x4, Grid B: (4,0)-(7,3) = 4x4
            Compute: fp32=False, math_approx=True, HiFi4 (all phases)
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        cores_a = _cores(0, 0, 3, 3)  # 4x4 = 16
        cores_b = _cores(4, 0, 7, 3)  # 4x4 = 16
        hidden = 128
        mm_cfg = _mm_config(grid_x=4, grid_y=4, in0_block_w=4, per_core_M=1, per_core_N=4)

        torch_a = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_B = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        # ---- Build + launch fused ----
        build_start = time.perf_counter()
        ta = _tt(torch_a, device)
        tb = _tt(torch_b, device)
        tw = _tt(torch_w, device)
        tbi = _tt(torch_bias, device)
        tB = _tt(torch_B, device)
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
        rb = rms_norm.rms_norm(
            tb,
            core_range_set=cores_b,
            weight=tw,
            epsilon=1e-5,
        )
        mb = matmul_desc(
            rb.output_tensors[0],
            tB,
            core_range_set=cores_b,
            program_config=mm_cfg,
            compute_kernel_config=_compute(),
        )
        fused = Parallel(Sequential(la, ma), Sequential(rb, mb)).build(device)
        build_time = time.perf_counter() - build_start

        outputs = composite.launch([fused])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        # ---- Unfused path (4 sequential dispatches) ----
        ua1 = ttnn.layer_norm(ta, weight=tw, bias=tbi, epsilon=1e-5, compute_kernel_config=_compute())
        ua2 = ttnn.matmul(ua1, tB, program_config=mm_cfg, compute_kernel_config=_compute())
        ub1 = ttnn.rms_norm(tb, weight=tw, epsilon=1e-5)
        ub2 = ttnn.matmul(ub1, tB, program_config=mm_cfg, compute_kernel_config=_compute())

        unfused_a = ttnn.to_torch(ua2)
        unfused_b = ttnn.to_torch(ub2)

        # Golden
        golden_a = torch_layer_norm(torch_a.float(), torch_w.float(), torch_bias.float()) @ torch_B.float()
        golden_b = torch_rms_norm(torch_b.float(), torch_w.float()) @ torch_B.float()

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.97)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.97)
        passing_ua, pcc_ua = comp_pcc(golden_a, unfused_a, pcc=0.97)
        passing_ub, pcc_ub = comp_pcc(golden_b, unfused_b, pcc=0.97)

        if iteration == 2:
            print(f"\n{'='*60}")
            print(f"Demo 4: Two parallel chains (disjoint cores)")
            print(f"  Fused compile: {build_time*1000:.2f} ms")
            print(f"  PCC: chain_a={pcc_a:.6f}  chain_b={pcc_b:.6f}")
            print(f"  Unfused PCC: chain_a={pcc_ua:.6f}  chain_b={pcc_ub:.6f}")
            print(f"{'='*60}")

        assert passing_a, f"Chain A PCC: {pcc_a}"
        assert passing_b, f"Chain B PCC: {pcc_b}"
        assert passing_ua, f"Unfused Chain A PCC: {pcc_ua}"
        assert passing_ub, f"Unfused Chain B PCC: {pcc_ub}"

    # -----------------------------------------------------------------
    # Demo 3: Branching topology on full 8x8 grid (segment barriers)
    # -----------------------------------------------------------------

    def test_demo3_branching(self, device, iteration):
        """Branching graph with nested Sequential/Parallel on full 8x8 grid.

        Topology:
                       stem_rms (8x8 = 64 cores)
                            |
                  +---------+---------+
                  |                   |
               left_ln             right_mm
             (0,0)-(7,3)         (0,4)-(7,7)
              8x4 = 32            8x4 = 32
                  |
             +----+----+
             |         |
           ll_mm    lr_rms
         (0,0)-(3,3) (4,0)-(7,3)
          4x4 = 16    4x4 = 16

        At each branching point, a segment barrier synchronizes all cores in
        the parent group: every core sends noc_semaphore_inc to core 0, core 0
        waits for all arrivals, then multicasts the release semaphore to all
        cores. This test exercises segment barriers at 64-core and 32-core scale.

        Setup:
            Input:  (1, 1, 2048, 128) -- 64x4 tiles, BF16, DRAM interleaved
            B:      (1, 1, 128, 128) -- matmul weight
            Grid:   8x8 = 64 cores total
            Compute: fp32=False, math_approx=True, HiFi4 (all phases)
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm, layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # Core grids -- split by rows (y) so matmul grid_y divides N_tiles=4
        stem_cores = _cores(0, 0, 7, 7)  # 8x8 = 64
        left_cores = _cores(0, 0, 7, 3)  # 8x4 = 32
        right_cores = _cores(0, 4, 7, 7)  # 8x4 = 32
        ll_cores = _cores(0, 0, 3, 3)  # 4x4 = 16
        lr_cores = _cores(4, 0, 7, 3)  # 4x4 = 16

        hidden = 128
        mm_compute = _compute()
        # right_mm: 8x4=32 cores, per_core_M=64/32=2, per_core_N=4 (N not distributed)
        right_mm_cfg = _mm_config(grid_x=8, grid_y=4, in0_block_w=4, per_core_M=2, per_core_N=4)
        # ll_mm: 4x4=16 cores, per_core_M=64/16=4, per_core_N=4 (N not distributed)
        ll_mm_cfg = _mm_config(grid_x=4, grid_y=4, in0_block_w=4, per_core_M=4, per_core_N=4)

        torch_input = torch.randn(1, 1, 2048, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_B = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        # ---- Build + launch fused ----
        build_start = time.perf_counter()
        tt_input = _tt(torch_input, device)
        tt_w = _tt(torch_w, device)
        tt_B = _tt(torch_B, device)
        s = rms_norm.rms_norm(tt_input, core_range_set=stem_cores, weight=tt_w, epsilon=1e-5)
        ll = layer_norm.layer_norm(
            s.output_tensors[0],
            core_range_set=left_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=mm_compute,
        )
        rm = matmul_desc(
            s.output_tensors[0],
            tt_B,
            core_range_set=right_cores,
            program_config=right_mm_cfg,
            compute_kernel_config=mm_compute,
        )
        llm = matmul_desc(
            ll.output_tensors[0],
            tt_B,
            core_range_set=ll_cores,
            program_config=ll_mm_cfg,
            compute_kernel_config=mm_compute,
        )
        lrr = rms_norm.rms_norm(
            ll.output_tensors[0],
            core_range_set=lr_cores,
            weight=tt_w,
            epsilon=1e-5,
        )
        fused = Sequential(
            s,
            Parallel(Sequential(ll, Parallel(llm, lrr)), rm),
        ).build(device)
        build_time = time.perf_counter() - build_start

        composite.launch([fused])

        # Reference outputs through the original op descriptors.
        result_ll = ttnn.to_torch(llm.output_tensors[0])
        result_lr = ttnn.to_torch(lrr.output_tensors[0])
        result_r = ttnn.to_torch(rm.output_tensors[0])

        # ---- Unfused path (5 sequential ttnn dispatches matching tree structure) ----
        # Matmul configs match fused branch core counts:
        #   ll_mm:    4x4 = 16 cores (matches fused ll_cores)
        #   right_mm: 8x4 = 32 cores (matches fused right_cores)
        # Norm ops use the default interleaved grid (64 cores) because the ttnn
        # API does not expose a core_range parameter for interleaved inputs.
        # This makes the unfused norm kernel times a lower bound (more cores = faster).
        u_stem = ttnn.rms_norm(tt_input, weight=tt_w, epsilon=1e-5)
        u_left = ttnn.layer_norm(u_stem, weight=tt_w, epsilon=1e-5)
        u_ll = ttnn.matmul(u_left, tt_B, program_config=ll_mm_cfg, compute_kernel_config=mm_compute)
        u_lr = ttnn.rms_norm(u_left, weight=tt_w, epsilon=1e-5)
        u_right = ttnn.matmul(u_stem, tt_B, program_config=right_mm_cfg, compute_kernel_config=mm_compute)

        u_ll_result = ttnn.to_torch(u_ll)
        u_lr_result = ttnn.to_torch(u_lr)
        u_right_result = ttnn.to_torch(u_right)

        # ---- Golden ----
        stem_golden = torch_rms_norm(torch_input.float(), torch_w.float())
        left_ln_golden = torch_layer_norm(stem_golden, torch_w.float())
        ll_mm_golden = left_ln_golden @ torch_B.float()
        lr_rms_golden = torch_rms_norm(left_ln_golden, torch_w.float())
        right_mm_golden = stem_golden @ torch_B.float()

        passing_ll, pcc_ll = comp_pcc(ll_mm_golden, result_ll, pcc=0.97)
        passing_lr, pcc_lr = comp_pcc(lr_rms_golden, result_lr, pcc=0.97)
        passing_r, pcc_r = comp_pcc(right_mm_golden, result_r, pcc=0.97)
        passing_ull, pcc_ull = comp_pcc(ll_mm_golden, u_ll_result, pcc=0.97)
        passing_ulr, pcc_ulr = comp_pcc(lr_rms_golden, u_lr_result, pcc=0.97)
        passing_ur, pcc_ur = comp_pcc(right_mm_golden, u_right_result, pcc=0.97)

        if iteration == 2:
            print(f"\n{'='*60}")
            print(f"Demo 3: Branching (8x8 grid = 64 cores)")
            print(f"  Fused compile: {build_time*1000:.2f} ms")
            print(f"  PCC: ll_mm={pcc_ll:.6f}  lr_rms={pcc_lr:.6f}  right_mm={pcc_r:.6f}")
            print(f"  Unfused PCC: ll_mm={pcc_ull:.6f}  lr_rms={pcc_ulr:.6f}  right_mm={pcc_ur:.6f}")
            print(f"{'='*60}")

        assert passing_ll, f"ll_mm PCC: {pcc_ll}"
        assert passing_lr, f"lr_rms PCC: {pcc_lr}"
        assert passing_r, f"right_mm PCC: {pcc_r}"
        assert passing_ull, f"Unfused ll_mm PCC: {pcc_ull}"
        assert passing_ulr, f"Unfused lr_rms PCC: {pcc_ulr}"
        assert passing_ur, f"Unfused right_mm PCC: {pcc_ur}"

    # -----------------------------------------------------------------
    # Demo 5: GlobalCircularBuffer mid-kernel write
    # -----------------------------------------------------------------

    def test_demo5_global_cb_mid_kernel(self, device, iteration):
        """Mid-kernel data exfiltration via GlobalCircularBuffer.

        Architecture:
            Sender core (0,0):
                Phase 0: DRAM(input_a) -> compute -> GlobalCB push
                Phase 1: DRAM(input_b) -> compute -> DRAM(output_b)
            Receiver core (1,0):
                GlobalCB -> DRAM(output_recv)

        The receiver gets data from Phase 0 while the sender continues
        with Phase 1. This demonstrates mid-kernel communication -- the
        fused kernel doesn't have to finish all phases before data reaches
        the consumer.

        Setup:
            Tiles: 8 tiles of 32x32 BF16
            Sender: core (0,0)
            Receiver: core (1,0)
            GlobalCB: 2-tile double-buffer

        Verification:
            output_recv == input_a  (GlobalCB transfer from Phase 0)
            output_b   == input_b  (Phase 1 identity copy)
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        num_tiles = 8
        shape = [1, 1, 32, 32 * num_tiles]

        torch_input_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_input_b = torch.randn(shape, dtype=torch.bfloat16)

        sender_core = ttnn.CoreCoord(0, 0)
        sender_range = _cores(0, 0, 0, 0)
        receiver_range = _cores(1, 0, 1, 0)

        gcb_size = TILE_SIZE_BF16 * 2
        gcb = ttnn.create_global_circular_buffer(
            device,
            [(sender_core, receiver_range)],
            gcb_size,
        )

        tia = _tt(torch_input_a, device)
        tib = _tt(torch_input_b, device)
        tt_output_b = _tt(torch.zeros(shape, dtype=torch.bfloat16), device)
        tt_output_recv = _tt(torch.zeros(shape, dtype=torch.bfloat16), device)
        oa = _build_globalcb_sender_op(tia, sender_range, gcb, num_tiles)
        ob = _build_identity_op(tib, tt_output_b, sender_range, num_tiles)
        con = _build_globalcb_consumer_op(tt_output_recv, receiver_range, gcb, num_tiles)

        build_start = time.perf_counter()
        fused = Parallel(Sequential(oa, ob), con).build(device)
        build_time = time.perf_counter() - build_start

        composite.launch([fused])

        result_recv = ttnn.to_torch(tt_output_recv)
        result_b = ttnn.to_torch(tt_output_b)

        passing_recv, pcc_recv = comp_pcc(torch_input_a, result_recv, pcc=0.999)
        passing_b, pcc_b = comp_pcc(torch_input_b, result_b, pcc=0.999)

        if iteration == 2:
            print(f"\n{'='*60}")
            print(f"Demo 5: GlobalCB mid-kernel write")
            print(f"  Fused compile: {build_time*1000:.2f} ms")
            print(f"  PCC: receiver={pcc_recv:.6f}  phase1={pcc_b:.6f}")
            print(f"{'='*60}")

        assert passing_recv, f"Receiver PCC: {pcc_recv}"
        assert passing_b, f"Phase 1 PCC: {pcc_b}"
