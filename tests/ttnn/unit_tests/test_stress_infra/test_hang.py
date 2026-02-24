# Tests that hang — dispatches a kernel that waits on an empty CB
import torch
import pytest
import ttnn


@pytest.mark.parametrize("dummy", [1])
def test_kernel_hang(device, dummy):
    """Dispatches a reader kernel that does cb_wait_front(0, 1) with no producer.
    This will hang until the dispatch timeout fires."""
    shape = [1, 1, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    io_tensors = [input_tensor, output_tensor]

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    cb_format = ttnn.CBFormatDescriptor(
        buffer_index=0,
        data_format=ttnn.bfloat16,
        page_size=2 * 1024,
    )
    cb_descriptor = ttnn.CBDescriptor(
        total_size=2 * 2 * 1024,
        core_ranges=core_grid,
        format_descriptors=[cb_format],
    )

    # This kernel does cb_wait_front(0, 1) — no producer, so it hangs forever
    hang_kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/test_stress_infra/kernels/hang_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[hang_kernel],
        semaphores=[],
        cbs=[cb_descriptor],
    )

    # This will hang — dispatch timeout should catch it
    ttnn.generic_op(io_tensors, program_descriptor)

    # Should never reach here
    assert False, "Should have timed out"
