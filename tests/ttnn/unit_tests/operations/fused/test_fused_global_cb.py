# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for fused ops + GlobalCircularBuffer communication.

Tests that a fused kernel (produced by Sequential) can push data through
a GlobalCircularBuffer to receiver cores running a separate consumer kernel.

Architecture:
    Sender core:  Sequential(identity_op, globalcb_sender_op).build()
    Receiver core: consumer op (waits on GlobalCB, writes to DRAM)
    Both run in a single program via composite.launch()
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion import Sequential, Parallel
from models.experimental.ops.descriptors import composite


# =============================================================================
# Inline C++ Kernel Sources
# =============================================================================

# Reader (riscv_0): Read tiles from interleaved DRAM into a local CB
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

# Compute: tile copy from input CB to output CB
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

# Writer (riscv_1): Write tiles from local CB to interleaved DRAM
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

# Writer (riscv_1): Push tiles from local CB to remote CB via GlobalCB
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

# Reader (riscv_0): Wait on GlobalCB, align local CB, push for writer
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

# Writer on receiver side: same as DRAM_WRITER_SOURCE but with cb_in named arg
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
# OpDescriptor Builders
# =============================================================================

TILE_SIZE_BF16 = 2048  # 32x32 * 2 bytes


def _make_cb_desc(buffer_index, core_ranges, total_size=TILE_SIZE_BF16, is_remote=False, gcb=None):
    """Create a CBDescriptor with a single format descriptor."""
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
    """Create a KernelDescriptor with SOURCE_CODE type."""
    k = ttnn.KernelDescriptor()
    k.kernel_source = source
    k.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    k.core_ranges = core_ranges
    k.named_compile_time_args = named_ct_args
    k.runtime_args = rt_args_per_core
    k.config = config
    return k


def build_identity_op(input_tensor, output_tensor, core_ranges, num_tiles):
    """Build an identity copy OpDescriptor: DRAM → compute → DRAM."""
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    # CBs: c_0 (input), c_4 (output)
    cb_in = _make_cb_desc(0, core_ranges)
    cb_out = _make_cb_desc(4, core_ranges)

    # Get core coord for runtime args
    coords = _get_core_coords(core_ranges)

    # Reader kernel
    reader = _make_kernel_desc(
        DRAM_READER_SOURCE,
        core_ranges,
        ttnn.ReaderConfigDescriptor(),
        [("cb_in", 0)],
        [(c, [src_addr, num_tiles]) for c in coords],
    )

    # Compute kernel
    compute = _make_kernel_desc(
        TILE_COPY_COMPUTE_SOURCE,
        core_ranges,
        ttnn.ComputeConfigDescriptor(),
        [("cb_in", 0), ("cb_out", 4)],
        [(c, [num_tiles]) for c in coords],
    )

    # Writer kernel
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


def build_globalcb_sender_op(input_tensor, core_ranges, gcb, num_tiles):
    """Build a GlobalCB sender OpDescriptor: DRAM → compute → remote CB push."""
    src_addr = input_tensor.buffer_address()

    # CBs: c_0 (input from DRAM), c_4 (output for compute), c_31 (remote GlobalCB)
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

    # No output_tensors — data goes to GlobalCB, not DRAM
    return OpDescriptor(
        descriptor=desc,
        input_tensors=[input_tensor],
        output_tensors=[],
        name="globalcb_sender",
    )


def build_globalcb_consumer_op(output_tensor, core_ranges, gcb, num_tiles):
    """Build a GlobalCB consumer OpDescriptor: remote CB → local CB → DRAM."""
    dst_addr = output_tensor.buffer_address()

    # Single CB with BOTH local (c_0) and remote (c_31) format descriptors,
    # linked to the GlobalCB.  This matches the reference pattern in benchmark 11:
    # the firmware's ALIGN_LOCAL_CBS_TO_REMOTE_CBS code needs both local and remote
    # indices on the same CB to auto-align the local CB to the GlobalCB's memory.
    cb_recv = ttnn.CBDescriptor()
    cb_recv.total_size = gcb.size()
    cb_recv.core_ranges = core_ranges
    local_fmt = ttnn.CBFormatDescriptor(
        buffer_index=0,
        data_format=ttnn.DataType.BFLOAT16,
        page_size=TILE_SIZE_BF16,
    )
    remote_fmt = ttnn.CBFormatDescriptor(
        buffer_index=31,
        data_format=ttnn.DataType.BFLOAT16,
        page_size=TILE_SIZE_BF16,
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

    return OpDescriptor(
        descriptor=desc,
        input_tensors=[],
        output_tensors=[output_tensor],
        name="globalcb_consumer",
    )


def _get_core_coords(core_ranges):
    """Extract ordered CoreCoord list from a CoreRangeSet."""
    coords = []
    for cr in core_ranges.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.append(ttnn.CoreCoord(x, y))
    return coords


# =============================================================================
# Tests
# =============================================================================


def build_direct_sender_op(input_tensor, core_ranges, gcb, num_tiles):
    """Build a direct sender op (no intermediate): DRAM → compute → remote CB push."""
    src_addr = input_tensor.buffer_address()

    # CBs: c_0 (input from DRAM), c_4 (output for compute), c_31 (remote GlobalCB)
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

    return OpDescriptor(
        descriptor=desc,
        input_tensors=[input_tensor],
        output_tensors=[],
        name="direct_sender",
    )


class TestFusedGlobalCB:
    """Test fused ops with GlobalCircularBuffer communication."""

    @pytest.mark.parametrize("num_tiles", [1, 8])
    def test_unfused_globalcb(self, device, num_tiles):
        """Non-fused: direct sender + consumer via GlobalCB.

        Validates GlobalCB communication works at all before testing fusion.
        """
        torch.manual_seed(42)
        shape = [1, 1, 32, 32 * num_tiles]
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(1, 0)
        sender_range = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})
        receiver_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})

        gcb_size = TILE_SIZE_BF16 * 2
        gcb = ttnn.create_global_circular_buffer(device, [(sender_core, receiver_range)], gcb_size)

        sender = build_direct_sender_op(tt_input, sender_range, gcb, num_tiles)
        consumer = build_globalcb_consumer_op(tt_output, receiver_range, gcb, num_tiles)

        fused = Parallel(sender, consumer).build(device)
        composite.launch([fused])

        result = ttnn.to_torch(tt_output)
        passing, pcc = comp_pcc(torch_input, result, pcc=0.999)
        assert passing, f"PCC mismatch: {pcc}"

    @pytest.mark.parametrize("num_tiles", [1, 8])
    def test_fused_identity_globalcb(self, device, num_tiles):
        """Fused identity(OpA) + GlobalCB sender(OpB) → receiver consumer.

        Tests the full pipeline:
        1. Sequential(identity, globalcb_sender).build() fuses two phases
        2. Consumer on receiver cores reads from GlobalCB
        3. Both run in one program via composite.launch()
        4. Verify receiver output == original input
        """
        torch.manual_seed(42)

        # Shape: num_tiles tiles of 32x32 BF16
        shape = [1, 1, 32, 32 * num_tiles]
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        # Create tensors on device
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_intermediate = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Core setup
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(1, 0)
        sender_range = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})
        receiver_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})

        # GlobalCircularBuffer: sender → receiver
        gcb_size = TILE_SIZE_BF16 * 2  # Double-buffer
        gcb = ttnn.create_global_circular_buffer(device, [(sender_core, receiver_range)], gcb_size)

        # Build OpDescriptors
        op_a = build_identity_op(tt_input, tt_intermediate, sender_range, num_tiles)
        op_b = build_globalcb_sender_op(tt_intermediate, sender_range, gcb, num_tiles)
        consumer = build_globalcb_consumer_op(tt_output, receiver_range, gcb, num_tiles)

        # Fuse sender chain + consumer into one program
        fused = Parallel(Sequential(op_a, op_b), consumer).build(device)
        composite.launch([fused])

        # Verify: receiver output should match original input
        result = ttnn.to_torch(tt_output)
        passing, pcc = comp_pcc(torch_input, result, pcc=0.999)
        assert passing, f"PCC mismatch: {pcc}"

    @pytest.mark.parametrize("num_tiles", [1, 8])
    def test_fused_mid_kernel_globalcb(self, device, num_tiles):
        """GlobalCB push in Phase 0, then fused kernel continues with Phase 1.

        Architecture:
            Phase 0 (OpA): DRAM(input_a) → compute → GlobalCB push (c_31)
            Phase 1 (OpB): DRAM(input_b) → compute → DRAM(output_b)
            Consumer:      GlobalCB(c_31) → DRAM(output_recv)

        The receiver gets data mid-kernel — Phase 0 pushes to the GlobalCB,
        the barrier fires, and Phase 1 starts processing completely different
        data while the receiver is still draining the GlobalCB.

        Verifies:
            - output_recv == input_a  (GlobalCB transfer from Phase 0)
            - output_b   == input_b  (Phase 1 identity, runs after barrier)
        """
        torch.manual_seed(42)
        shape = [1, 1, 32, 32 * num_tiles]
        torch_input_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_input_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_input_a = ttnn.from_torch(
            torch_input_a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input_b = ttnn.from_torch(
            torch_input_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output_b = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output_recv = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(1, 0)
        sender_range = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})
        receiver_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})

        gcb_size = TILE_SIZE_BF16 * 2
        gcb = ttnn.create_global_circular_buffer(device, [(sender_core, receiver_range)], gcb_size)

        # Phase 0: read input_a, push to GlobalCB
        op_a = build_globalcb_sender_op(tt_input_a, sender_range, gcb, num_tiles)
        # Phase 1: read input_b, write to output_b (completely independent data)
        op_b = build_identity_op(tt_input_b, tt_output_b, sender_range, num_tiles)
        # Receiver: drain GlobalCB → output_recv
        consumer = build_globalcb_consumer_op(tt_output_recv, receiver_range, gcb, num_tiles)

        # Fuse everything: Sequential(sender→identity) || consumer
        fused = Parallel(Sequential(op_a, op_b), consumer).build(device)
        composite.launch([fused])

        # Receiver got input_a via GlobalCB from Phase 0
        result_recv = ttnn.to_torch(tt_output_recv)
        passing_recv, pcc_recv = comp_pcc(torch_input_a, result_recv, pcc=0.999)
        assert passing_recv, f"Receiver PCC mismatch: {pcc_recv}"

        # Phase 1 identity output matches input_b
        result_b = ttnn.to_torch(tt_output_b)
        passing_b, pcc_b = comp_pcc(torch_input_b, result_b, pcc=0.999)
        assert passing_b, f"Phase 1 PCC mismatch: {pcc_b}"
