# Test that triggers NoC sanitizer — misaligned address read
# Requires watcher with NoC sanitizer enabled (--dev mode)
import torch
import pytest
import ttnn


@pytest.mark.parametrize("dummy", [1])
def test_noc_misaligned_read(device, dummy):
    """Dispatches a kernel that reads from a misaligned NoC address.
    With watcher NoC sanitizer enabled, this should be caught."""
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

    # Runtime args: pass the input tensor's buffer address so the kernel can
    # offset it by 7 bytes to create a misaligned read
    rt_args = ttnn.RuntimeArgs()
    rt_args[0][0] = [input_tensor.buffer_address()]

    misaligned_kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/test_stress_infra/kernels/noc_misaligned_read.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[],
        runtime_args=rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[misaligned_kernel],
        semaphores=[],
        cbs=[cb_descriptor],
    )

    # This should trigger NoC sanitizer in dev mode, or just corrupt data silently without it
    ttnn.generic_op(io_tensors, program_descriptor)
