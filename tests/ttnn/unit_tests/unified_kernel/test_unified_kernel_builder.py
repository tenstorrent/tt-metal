# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test UnifiedKernelBuilder following the pattern from universal-kernel-diff.patch.

Tests the Python wrapper for UnifiedKernelConfigBuilder, validating:
- INIT_ARGUMENTS generation
- Buffer handling with TensorAccessorArgs
- Config type generation (Reader/Writer/Compute)
- Generated tile constants
- Runtime argument tracking
"""

import pytest
import torch
import ttnn
from ttnn.unified_kernel import UnifiedKernelBuilder, McastGroup, Role, BufferMode


def test_unified_kernel_builder_basic_execute(device):
    """Test basic execution with generic_op - exp function."""
    # Create test tensors
    shape = [1, 1, 32, 32]  # Single tile
    num_tiles = 1
    py_tensor_in = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_in = ttnn.from_torch(
        py_tensor_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_tensor_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # Set up core grid
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    # Use existing kernels that work (following test_generic_op.py pattern)
    in_cb = 0
    out_cb = 16
    cb_page_size = 2 * 1024  # bfloat16 tile size
    cb_total_size = 2 * cb_page_size  # double buffer

    # Create CBs
    in_cb_format = ttnn.CBFormatDescriptor(buffer_index=in_cb, data_format=ttnn.bfloat16, page_size=cb_page_size)
    out_cb_format = ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=ttnn.bfloat16, page_size=cb_page_size)
    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size, core_ranges=core_grid, format_descriptors=[in_cb_format]
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size, core_ranges=core_grid, format_descriptors=[out_cb_format]
    )

    # Get compile-time args from TensorAccessorArgs
    reader_compile_time_args = ttnn.TensorAccessorArgs(tt_tensor_in).get_compile_time_args()
    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(tt_tensor_out).get_compile_time_args())
    compute_compile_time_args = [num_tiles, 1]  # [per_core_tiles, pop_count]

    # Create per-core runtime args
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [tt_tensor_in.buffer_address(), num_tiles, 0]
    writer_rt_args[0][0] = [tt_tensor_out.buffer_address(), num_tiles, 0]

    # Reader kernel
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer kernel
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel - exp
    sfpu_defines = [("SFPU_OP_EXP_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/eltwise_sfpu.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    # Execute the program
    io_tensors = [tt_tensor_in, tt_tensor_out]
    output = ttnn.generic_op(io_tensors, program)

    # Validate the result - should be exp(input)
    torch_output = ttnn.to_torch(output)
    expected = torch.exp(py_tensor_in)

    assert torch.allclose(
        torch_output, expected, rtol=0.01, atol=0.01
    ), f"Output mismatch: max diff = {(torch_output - expected).abs().max()}"


def test_unified_kernel_builder_unicast(device):
    """Test UnifiedKernelBuilder for unicast operation (one sender to one receiver)."""
    # Create test tensors
    shape = [1, 1, 32, 32]  # Single tile
    py_tensor = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_sender = ttnn.from_torch(py_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_tensor_receiver = ttnn.from_torch(
        torch.zeros(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Create kernel source for unicast
    kernel_source = """
#include "unified_kernel/unified_kernel_api.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    // Read tile into source CB first (for sender)
    if constexpr (MY_ROLE == ROLE_MCAST_SENDER) {
        uint32_t cb_id = GET_CB_ID(in0);
        cb_reserve_back(cb_id, 1);
        uint32_t tile_addr = get_write_ptr(cb_id);
        noc_async_read_tile(0, in0, tile_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }

    // Broadcast tile using MPI-style collective primitive
    bcast_tile(receivers, in0, out, 0);
}
"""

    # Set up core grid - sender and receiver cores
    sender_core = ttnn.CoreCoord(0, 0)
    receiver_core = ttnn.CoreCoord(0, 1)

    sender_range = ttnn.CoreRange(sender_core, sender_core)
    receiver_range = ttnn.CoreRange(receiver_core, receiver_core)
    receiver_grid = ttnn.CoreRangeSet([receiver_range])

    # Build unified kernel program with unicast group
    builder = (
        UnifiedKernelBuilder(kernel_source)
        .add_tensor("in0", tt_tensor_sender)
        .add_tensor("out", tt_tensor_receiver)
        .add_mcast_group(
            "receivers",
            receivers=receiver_grid,
            sender=sender_core,
            noc=ttnn.NOC.NOC_0,
        )
        .set_core_grid(ttnn.CoreRangeSet([sender_range, receiver_range]))
    )

    program = builder.build(device)

    # Validate ProgramDescriptor structure
    assert program is not None
    # Should have sender and receiver kernels
    assert len(program.kernels) == 2, "Should have 2 kernels (sender, receiver)"
    assert len(program.cbs) == 2, "Should have 2 circular buffers (in0, out)"
    assert len(program.semaphores) == 2, "Should have 2 semaphores (sender sem, receiver sem)"

    # Validate kernel configs - should have role defines
    sender_kernel = None
    receiver_kernel = None

    for kernel in program.kernels:
        defines_dict = dict(kernel.defines)
        if "MCAST_SENDER" in defines_dict:
            sender_kernel = kernel
        elif "MCAST_RECEIVER" in defines_dict:
            receiver_kernel = kernel

    assert sender_kernel is not None, "Should have sender kernel"
    assert receiver_kernel is not None, "Should have receiver kernel"

    # Check role defines
    sender_defines = dict(sender_kernel.defines)
    receiver_defines = dict(receiver_kernel.defines)

    assert sender_defines.get("MY_ROLE") == str(Role.MCAST_SENDER.value), "Sender should have MCAST_SENDER role"
    assert sender_defines.get("MCAST_SENDER") == "1", "Sender should have MCAST_SENDER define"
    assert receiver_defines.get("MY_ROLE") == str(Role.MCAST_RECEIVER.value), "Receiver should have MCAST_RECEIVER role"
    assert receiver_defines.get("MCAST_RECEIVER") == "1", "Receiver should have MCAST_RECEIVER define"

    # Check that sender has named compile-time args for multicast
    assert len(sender_kernel.named_compile_time_args) > 0, "Sender should have named compile-time args"
    named_args_dict = dict(sender_kernel.named_compile_time_args)
    assert "receivers_num_cores" in named_args_dict, "Should have num_cores arg"
    assert "receivers_data_sender_semaphore" in named_args_dict, "Should have sender semaphore arg"
    assert "receivers_data_receiver_semaphore" in named_args_dict, "Should have receiver semaphore arg"


def test_unified_kernel_builder_multicast(device):
    """Test UnifiedKernelBuilder for multicast operation (one sender to multiple receivers)."""
    # Create test tensors
    shape = [1, 1, 32, 32]  # Single tile
    py_tensor = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_sender = ttnn.from_torch(py_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_tensor_receiver = ttnn.from_torch(
        torch.zeros(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Create kernel source for multicast
    kernel_source = """
#include "unified_kernel/unified_kernel_api.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    // Read tile into source CB first (for sender)
    if constexpr (MY_ROLE == ROLE_MCAST_SENDER) {
        uint32_t cb_id = GET_CB_ID(in0);
        cb_reserve_back(cb_id, 1);
        uint32_t tile_addr = get_write_ptr(cb_id);
        noc_async_read_tile(0, in0, tile_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }

    // Broadcast tile using MPI-style collective primitive
    bcast_tile(receivers, in0, out, 0);
}
"""

    # Set up core grid - sender and multiple receivers
    sender_core = ttnn.CoreCoord(0, 0)
    receiver_start = ttnn.CoreCoord(0, 1)
    receiver_end = ttnn.CoreCoord(0, 3)  # 3 receivers

    sender_range = ttnn.CoreRange(sender_core, sender_core)
    receiver_range = ttnn.CoreRange(receiver_start, receiver_end)
    receiver_grid = ttnn.CoreRangeSet([receiver_range])

    # Build unified kernel program with multicast group
    builder = (
        UnifiedKernelBuilder(kernel_source)
        .add_tensor("in0", tt_tensor_sender)
        .add_tensor("out", tt_tensor_receiver)
        .add_mcast_group(
            "receivers",
            receivers=receiver_grid,
            sender=sender_core,
            noc=ttnn.NOC.NOC_0,
        )
        .set_core_grid(ttnn.CoreRangeSet([sender_range, receiver_range]))
    )

    program = builder.build(device)

    # Validate ProgramDescriptor structure
    assert program is not None
    # Should have 1 sender kernel and 1 receiver kernel (receiver kernel runs on multiple cores)
    assert len(program.kernels) == 2, "Should have 2 kernels (sender, receiver)"
    assert len(program.cbs) == 2, "Should have 2 circular buffers (in0, out)"
    assert len(program.semaphores) == 2, "Should have 2 semaphores (sender sem, receiver sem)"

    # Validate kernel configs - should have role defines
    sender_kernel = None
    receiver_kernel = None

    for kernel in program.kernels:
        defines_dict = dict(kernel.defines)
        if "MCAST_SENDER" in defines_dict:
            sender_kernel = kernel
        elif "MCAST_RECEIVER" in defines_dict:
            receiver_kernel = kernel

    assert sender_kernel is not None, "Should have sender kernel"
    assert receiver_kernel is not None, "Should have receiver kernel"

    # Check role defines
    sender_defines = dict(sender_kernel.defines)
    receiver_defines = dict(receiver_kernel.defines)

    assert sender_defines.get("MY_ROLE") == str(Role.MCAST_SENDER.value), "Sender should have MCAST_SENDER role"
    assert sender_defines.get("MCAST_SENDER") == "1", "Sender should have MCAST_SENDER define"
    assert receiver_defines.get("MY_ROLE") == str(Role.MCAST_RECEIVER.value), "Receiver should have MCAST_RECEIVER role"
    assert receiver_defines.get("MCAST_RECEIVER") == "1", "Receiver should have MCAST_RECEIVER define"

    # Check that sender has named compile-time args for multicast
    assert len(sender_kernel.named_compile_time_args) > 0, "Sender should have named compile-time args"
    named_args_dict = dict(sender_kernel.named_compile_time_args)
    assert "receivers_num_cores" in named_args_dict, "Should have num_cores arg"
    assert named_args_dict["receivers_num_cores"] == 3, "Should have 3 receivers"
    assert "receivers_data_sender_semaphore" in named_args_dict, "Should have sender semaphore arg"
    assert "receivers_data_receiver_semaphore" in named_args_dict, "Should have receiver semaphore arg"

    # Check that receiver kernel runs on multiple cores
    receiver_core_ranges = receiver_kernel.core_ranges
    receiver_cores = []
    for core_range in receiver_core_ranges.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                receiver_cores.append((x, y))

    assert len(receiver_cores) == 3, "Should have 3 receiver cores"


def test_unified_kernel_builder_with_generated_tile_constant(device):
    """Test UnifiedKernelBuilder with generated tile constants (like reduce_h example)."""
    # Create test tensor
    shape = [1, 1, 32, 32]
    py_tensor = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_in = ttnn.from_torch(py_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_tensor_out = ttnn.from_torch(
        torch.zeros(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Create kernel source (simplified reduce example)
    kernel_source = """
#include "unified_common.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    #ifdef COMPILE_FOR_TRISC
    compute_kernel_hw_startup(src0_cb, scaler_tile_cb, out_cb);
    reduce_init(src0_cb, scaler_tile_cb, out_cb);
    #endif

    // Simplified reduce logic would go here
}
"""

    # Set up core grid
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    # Pack scaler value (like in reduce_h example)
    # For simplicity, use a packed value directly (in real code this would pack bfloat16)
    packed_scaler = 0x3F800000  # Packed representation of 1.0 in bfloat16 format

    # Build unified kernel program with generated tile constant
    builder = (
        UnifiedKernelBuilder(kernel_source, math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_compile_time_arg("Ht", 1)
        .add_compile_time_arg("Wt", 1)
        .add_compile_time_arg("row_chunk", 1)
        .add_compile_time_arg("packed_scaler_value", packed_scaler)
        .add_runtime_arg("start_write_page_id", 0)
        .add_runtime_arg("col_start_tile_id", 0)
        .add_runtime_arg("curr_col_in_batch", 0)
        .add_tensor("src0", tt_tensor_in)
        .add_tensor("out", tt_tensor_out)
        .add_generated_tile_constant(
            name="scaler_tile",
            data_format=ttnn.bfloat16,
            generator_code="generate_reduce_scaler(scaler_tile_cb, packed_scaler_value)",
        )
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Validate structure
    assert program is not None
    assert len(program.kernels) == 3, "Should have 3 kernels"
    assert len(program.cbs) == 3, "Should have 3 CBs (src0, out, scaler_tile)"

    # Check that generated tile constant appears in INIT_ARGUMENTS
    reader_defines = dict(program.kernels[0].defines)
    compute_defines = dict(program.kernels[1].defines)

    reader_init = reader_defines["INIT_ARGUMENTS"]
    compute_init = compute_defines["INIT_ARGUMENTS"]

    assert "scaler_tile_cb" in reader_init, "Reader INIT_ARGUMENTS should have scaler_tile_cb"
    assert "scaler_tile" in reader_init, "Reader INIT_ARGUMENTS should have scaler_tile"
    assert "generate_reduce_scaler" in reader_init, "Reader should generate scaler tile"
    assert "cb_wait_front" in compute_init, "Compute should wait for scaler tile"


def test_unified_kernel_builder_runtime_arg_tracking(device):
    """Test runtime argument index tracking."""
    kernel_source = "KERNEL_MAIN { INIT_ARGUMENTS }"

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    builder = (
        UnifiedKernelBuilder(kernel_source)
        .add_runtime_arg("arg0", 10)
        .add_runtime_arg("arg1", 20)
        .add_runtime_arg("arg2", 30)
        .set_core_grid(core_grid)
    )

    # Test runtime arg index lookup
    assert builder.get_runtime_arg_idx("arg0") == 0
    assert builder.get_runtime_arg_idx("arg1") == 1
    assert builder.get_runtime_arg_idx("arg2") == 2

    # Test buffer addresses start index
    builder.add_buffer("in0", None, ttnn.bfloat16)
    assert builder.buffer_addresses_start_runtime_arg_idx() == 3

    # Test that invalid arg name raises error
    with pytest.raises(ValueError, match="Runtime argument 'invalid' not found"):
        builder.get_runtime_arg_idx("invalid")


def test_unified_kernel_builder_compile_time_args(device):
    """Test compile-time argument handling."""
    kernel_source = "KERNEL_MAIN { INIT_ARGUMENTS }"

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    builder = (
        UnifiedKernelBuilder(kernel_source)
        .add_compile_time_arg("Ht", 3)
        .add_compile_time_arg("Wt", 4)
        .add_compile_time_arg("NC", 1)
        .add_compile_time_arg("row_chunk", 2)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Check that compile-time args appear in INIT_ARGUMENTS
    reader_defines = dict(program.kernels[0].defines)
    init_args = reader_defines["INIT_ARGUMENTS"]

    assert "constexpr uint32_t Ht = 3" in init_args
    assert "constexpr uint32_t Wt = 4" in init_args
    assert "constexpr uint32_t NC = 1" in init_args
    assert "constexpr uint32_t row_chunk = 2" in init_args


def test_unified_kernel_builder_common_runtime_args(device):
    """Test common runtime argument handling."""
    kernel_source = "KERNEL_MAIN { INIT_ARGUMENTS }"

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    builder = (
        UnifiedKernelBuilder(kernel_source)
        .add_common_runtime_arg("common_arg0", 100)
        .add_common_runtime_arg("common_arg1", 200)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Check that common runtime args appear in INIT_ARGUMENTS
    reader_defines = dict(program.kernels[0].defines)
    init_args = reader_defines["INIT_ARGUMENTS"]

    assert "get_common_arg_val" in init_args
    assert "common_arg0" in init_args
    assert "common_arg1" in init_args

    # Check that common runtime args are set on kernels
    assert len(program.kernels[0].common_runtime_args) >= 0  # May be 0 if no TensorAccessorArgs use them


def test_unified_kernel_builder_buffer_mode(device):
    """Test buffer mode option for tensors."""
    shape = [1, 1, 32, 32]  # Single tile
    py_tensor = torch.rand(shape).to(torch.bfloat16)

    tt_tensor = ttnn.from_torch(py_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    kernel_source = "KERNEL_MAIN { INIT_ARGUMENTS }"

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    # Test with double buffer (default)
    builder_double = (
        UnifiedKernelBuilder(kernel_source)
        .add_tensor("in0", tt_tensor, mode=BufferMode.DOUBLE)
        .add_tensor("out", tt_tensor)  # default is DOUBLE
        .set_core_grid(core_grid)
    )
    program_double = builder_double.build(device)

    # Test with single buffer
    builder_single = (
        UnifiedKernelBuilder(kernel_source)
        .add_tensor("in0", tt_tensor, mode=BufferMode.SINGLE)
        .add_tensor("out", tt_tensor, mode=BufferMode.SINGLE)
        .set_core_grid(core_grid)
    )
    program_single = builder_single.build(device)

    # Both should have 2 CBs
    assert len(program_double.cbs) == 2
    assert len(program_single.cbs) == 2

    # Double buffered CBs should have 2x the total_size of single buffered
    double_cb_size = program_double.cbs[0].total_size
    single_cb_size = program_single.cbs[0].total_size
    assert (
        double_cb_size == 2 * single_cb_size
    ), f"Double buffer should be 2x single: {double_cb_size} vs {single_cb_size}"


def test_unified_kernel_add_tiles(device):
    """Test UnifiedKernelBuilder with a true unified kernel that does A+B.

    This test uses the UnifiedKernelBuilder to generate a ProgramDescriptor
    from a single unified kernel source that includes unified_common.h
    and uses KERNEL_MAIN, INIT_ARGUMENTS, read_tile, write_tile macros.
    """
    # Create test tensors - 2 tiles
    shape = [1, 1, 32, 64]  # 2 tiles
    num_tiles = 2

    py_tensor_a = torch.rand(shape).to(torch.bfloat16)
    py_tensor_b = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_a = ttnn.from_torch(
        py_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_tensor_b = ttnn.from_torch(
        py_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_tensor_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # The unified kernel source - ONE file compiled 3 ways!
    # NO #ifdefs needed - unified_common.h handles all processor differences!
    unified_kernel_source = """
#include "unified_common.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    // Initialize for binary add (no-op on data movement processors)
    INIT_BINARY_ADD(in0_cb, in1_cb, out_cb);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Read tiles (reader reads from DRAM, compute waits on CB)
        auto tile0 = read_tile(in0, i);
        auto tile1 = read_tile(in1, i);

        // Compute A+B (no-op on data movement processors)
        ACQUIRE_DST();
        add_tiles(tile0, tile1, 0);

        // Write result (compute packs to CB, writer writes to DRAM)
        // write_tile automatically waits for DST if needed
        write_tile(0, out, i);

        RELEASE_DST();
    }
}
"""

    # Set up core grid
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    # Build unified kernel program using the builder
    # Note: n_tiles is automatically computed from tensor shape!
    builder = (
        UnifiedKernelBuilder(unified_kernel_source, math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_tensor("in0", tt_tensor_a)
        .add_tensor("in1", tt_tensor_b)
        .add_tensor("out", tt_tensor_out)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Validate the structure
    assert program is not None
    assert len(program.kernels) == 3, f"Should have 3 kernels (reader, compute, writer), got {len(program.kernels)}"
    assert len(program.cbs) == 3, f"Should have 3 CBs (in0, in1, out), got {len(program.cbs)}"

    # Check that all kernels have the same source (unified!)
    kernel_sources = [k.kernel_source for k in program.kernels]
    assert (
        kernel_sources[0] == kernel_sources[1] == kernel_sources[2]
    ), "All 3 kernels should have the same source code (unified kernel)"

    # Check that INIT_ARGUMENTS is defined
    for kernel in program.kernels:
        defines = dict(kernel.defines)
        assert "INIT_ARGUMENTS" in defines, "Each kernel should have INIT_ARGUMENTS define"
        init_args = defines["INIT_ARGUMENTS"]
        assert "n_tiles" in init_args, "INIT_ARGUMENTS should contain n_tiles"
        assert "in0_cb" in init_args, "INIT_ARGUMENTS should contain in0_cb"
        assert "in1_cb" in init_args, "INIT_ARGUMENTS should contain in1_cb"
        assert "out_cb" in init_args, "INIT_ARGUMENTS should contain out_cb"

    # Check kernel configs
    reader_kernel = program.kernels[0]
    compute_kernel = program.kernels[1]
    writer_kernel = program.kernels[2]

    assert isinstance(reader_kernel.config, ttnn.ReaderConfigDescriptor), "First kernel should be reader"
    assert isinstance(compute_kernel.config, ttnn.ComputeConfigDescriptor), "Second kernel should be compute"
    assert isinstance(writer_kernel.config, ttnn.WriterConfigDescriptor), "Third kernel should be writer"

    # Execute the program!
    io_tensors = [tt_tensor_a, tt_tensor_b, tt_tensor_out]
    output = ttnn.generic_op(io_tensors, program)

    # Validate the result: A + B
    torch_output = ttnn.to_torch(output)
    expected = py_tensor_a + py_tensor_b

    assert torch.allclose(
        torch_output, expected, rtol=0.01, atol=0.01
    ), f"Output mismatch for A+B: max diff = {(torch_output - expected).abs().max()}"


def test_unified_kernel_mul_tiles(device):
    """Test A*B (element-wise multiply) using UnifiedKernelBuilder.

    This test uses a single unified kernel source that:
    1. Reads tiles from A, B
    2. Computes A*B using FPU mul_tiles
    3. Writes result to output
    """
    # Create test tensors - use multiple tiles
    shape = [1, 1, 32, 64]  # 2 tiles
    num_tiles = 2

    py_tensor_a = torch.rand(shape).to(torch.bfloat16)
    py_tensor_b = torch.rand(shape).to(torch.bfloat16)

    tt_tensor_a = ttnn.from_torch(
        py_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_tensor_b = ttnn.from_torch(
        py_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_tensor_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # Unified kernel source for A*B
    # NO #ifdefs needed - unified_common.h handles all processor differences!
    unified_kernel_source = """
#include "unified_common.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    // Initialize for binary multiply (no-op on data movement processors)
    INIT_BINARY_MUL(a_cb, b_cb, out_cb);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Read tiles from A and B
        auto tile_a = read_tile(a, i);
        auto tile_b = read_tile(b, i);

        // Compute A*B element-wise (no-op on data movement processors)
        ACQUIRE_DST();
        mul_tiles(tile_a, tile_b, 0);

        // Write result
        // write_tile automatically waits for DST if needed
        write_tile(0, out, i);

        RELEASE_DST();
    }
}
"""

    # Set up core grid
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreRangeSet([core_range])

    # Build unified kernel program using the builder
    # Note: n_tiles is automatically computed from tensor shape!
    builder = (
        UnifiedKernelBuilder(unified_kernel_source, math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_tensor("a", tt_tensor_a)
        .add_tensor("b", tt_tensor_b)
        .add_tensor("out", tt_tensor_out)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Execute the program
    io_tensors = [tt_tensor_a, tt_tensor_b, tt_tensor_out]
    output = ttnn.generic_op(io_tensors, program)

    # Validate the result: A*B
    torch_output = ttnn.to_torch(output)
    expected = py_tensor_a * py_tensor_b

    assert torch.allclose(
        torch_output, expected, rtol=0.02, atol=0.02
    ), f"Output mismatch for A*B: max diff = {(torch_output - expected).abs().max()}"


def test_unified_kernel_fma_16_tiles(device):
    """
    Test FMA with 16 tiles (4x4): A*B + C
    Key insight from ternary_addcmul_fpu.cpp: both mul_tiles_init and
    binary_dest_reuse_tiles_init must be called INSIDE the loop!
    """
    torch.manual_seed(42)

    # Create test tensors - 16 tiles (4x4)
    shape = (1, 1, 4 * 32, 4 * 32)  # 16 tiles total

    py_tensor_a = torch.randn(shape, dtype=torch.bfloat16)
    py_tensor_b = torch.randn(shape, dtype=torch.bfloat16)
    py_tensor_c = torch.randn(shape, dtype=torch.bfloat16)

    # Convert to ttnn tensors
    tt_tensor_a = ttnn.from_torch(py_tensor_a, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_b = ttnn.from_torch(py_tensor_b, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_c = ttnn.from_torch(py_tensor_c, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_out = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

    # Clean DstTile API - mul/add auto-reinit when switching ops
    # User only provides one-time binary_op_init_common
    unified_kernel_source = """
    // One-time common init
    binary_op_init_common(a_cb, b_cb, out_cb);

    for (uint32_t i = 0; i < a_n_tiles; i++) {
        auto tile_a = read_tile(a, i);
        auto tile_b = read_tile(b, i);
        auto tile_c = read_tile(c, i);

        ACQUIRE_DST();

        DstTile result = mul(tile_a, tile_b);
        result = add(result, tile_c);

        write_tile(result, out, i);
        RELEASE_DST();
    }
"""

    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    core_grid = ttnn.CoreRangeSet([core_range])

    # No need to pass total_tiles - n_tiles is auto-computed from tensors!
    builder = (
        UnifiedKernelBuilder(unified_kernel_source, math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_tensor("a", tt_tensor_a)
        .add_tensor("b", tt_tensor_b)
        .add_tensor("c", tt_tensor_c)
        .add_tensor("out", tt_tensor_out)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Execute the program
    io_tensors = [tt_tensor_a, tt_tensor_b, tt_tensor_c, tt_tensor_out]
    output = ttnn.generic_op(io_tensors, program)

    # Validate: A*B + C
    torch_output = ttnn.to_torch(output)
    expected = py_tensor_a * py_tensor_b + py_tensor_c

    assert torch.allclose(
        torch_output, expected, rtol=0.02, atol=0.02
    ), f"Output mismatch for 16-tile A*B+C: max diff = {(torch_output - expected).abs().max()}"


def test_unified_kernel_fma_multicore(device):
    """
    Test multicore FMA: A*B + C distributed across 2x2 = 4 cores.
    Each core processes its share of tiles using tile_range().
    """
    torch.manual_seed(42)

    # Create larger tensors - 64 tiles (8x8) for 4 cores = 16 tiles/core
    shape = (1, 1, 8 * 32, 8 * 32)  # 64 tiles total

    py_tensor_a = torch.randn(shape, dtype=torch.bfloat16)
    py_tensor_b = torch.randn(shape, dtype=torch.bfloat16)
    py_tensor_c = torch.randn(shape, dtype=torch.bfloat16)

    # Convert to ttnn tensors
    tt_tensor_a = ttnn.from_torch(py_tensor_a, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_b = ttnn.from_torch(py_tensor_b, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_c = ttnn.from_torch(py_tensor_c, device=device, layout=ttnn.TILE_LAYOUT)
    tt_tensor_out = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

    # Multicore kernel using tile_range() for work distribution
    # grid_x, grid_y, a_n_tiles are auto-generated in INIT_ARGUMENTS
    unified_kernel_source = """
    binary_op_init_common(a_cb, b_cb, out_cb);
    auto range = tile_range(a_n_tiles, grid_x, grid_y);

    for (uint32_t i = range.start; i < range.end; i++) {
        auto tile_a = read_tile(a, i);
        auto tile_b = read_tile(b, i);
        auto tile_c = read_tile(c, i);

        ACQUIRE_DST();
        DstTile result = mul(tile_a, tile_b);
        result = add(result, tile_c);
        write_tile(result, out, i);
        RELEASE_DST();
    }
"""

    # 2x2 grid = 4 cores
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
    core_grid = ttnn.CoreRangeSet([core_range])

    builder = (
        UnifiedKernelBuilder(unified_kernel_source, math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_tensor("a", tt_tensor_a)
        .add_tensor("b", tt_tensor_b)
        .add_tensor("c", tt_tensor_c)
        .add_tensor("out", tt_tensor_out)
        .set_core_grid(core_grid)
    )

    program = builder.build(device)

    # Execute the program
    io_tensors = [tt_tensor_a, tt_tensor_b, tt_tensor_c, tt_tensor_out]
    output = ttnn.generic_op(io_tensors, program)

    # Validate: A*B + C
    torch_output = ttnn.to_torch(output)
    expected = py_tensor_a * py_tensor_b + py_tensor_c

    assert torch.allclose(
        torch_output, expected, rtol=0.02, atol=0.02
    ), f"Output mismatch for multicore A*B+C: max diff = {(torch_output - expected).abs().max()}"
