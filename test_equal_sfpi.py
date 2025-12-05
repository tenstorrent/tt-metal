import os
import ttnn
import torch
import math
from jinja2 import Environment, FileSystemLoader


def generate_kernel_from_sfpi_source(kernel_name, sfpi_kernel_name):
    kernel_name = "unary-sfpi"

    jinja_env = Environment(
        loader=FileSystemLoader("templates"),
    )

    sfpi_kernel_code = ""
    with open(f"templates/sfpi/{sfpi_kernel_name}.cpp", "r") as f:
        sfpi_kernel_code = f.read()

    template = jinja_env.get_template(f"{kernel_name}.cpp.j2")

    kernel_source_code = template.render(
        SFPU_KERNEL_NAME=f"calculate_sfpi_kernel",
        SFPU_KERNEL_IMPL=sfpi_kernel_code,
    )

    return kernel_source_code


def generic_unary_kernel(
    compute_kernel_source_code, ttnn_input_tensor, ttnn_output_tensor=None, core_grid=None, metal_home_dir=None
):
    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)])

    if metal_home_dir is None:
        metal_home_dir = os.getenv("TT_METAL_HOME")
        if metal_home_dir is None:
            raise RuntimeError("TT_METAL_HOME environment variable is not set")

    assert ttnn_output_tensor is not None

    io_tensors = [ttnn_input_tensor, ttnn_output_tensor]

    if core_grid is None:
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    input_cb_data_format = ttnn_input_tensor.dtype  # this will be mapped tt::DataFormat::Float16_b

    if input_cb_data_format == ttnn.float32:
        bytes_per_datum = 4
    else:
        bytes_per_datum = 2

    cb_total_size = 2 * bytes_per_datum * 1024  # tt::DataFormat::Float16_b hard coded to have size 2 * 1024
    cb_page_size = bytes_per_datum * 1024

    in_cb = 0
    out_cb = 1
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    # tile_shape = ttnn_input_tensor.get_tile().tile_shape
    # tile_volume = tile_shape[0] * tile_shape[1]
    # tensor_volume = math.prod(ttnn_input_tensor.shape)
    # num_tiles = tensor_volume // tile_volume

    tile_volume = math.prod(ttnn_input_tensor.get_tile().tile_shape)
    padded_volume = math.prod(ttnn_input_tensor.padded_shape)
    num_tiles = padded_volume // tile_volume

    reader_compile_time_args = ttnn.TensorAccessorArgs(ttnn_input_tensor).get_compile_time_args()
    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(ttnn_output_tensor).get_compile_time_args())
    compute_compile_time_args = [num_tiles, 1]
    reader_rt_args = [ttnn_input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = [ttnn_output_tensor.buffer_address(), num_tiles, 0]

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=[[reader_rt_args]],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=[[writer_rt_args]],
        config=ttnn.WriterConfigDescriptor(),
    )

    sfpu_defines = []

    compute_kernel_config = ttnn.ComputeConfigDescriptor()
    compute_kernel_config.fp32_dest_acc_en = ttnn_input_tensor.dtype == ttnn.float32
    compute_kernel_config.math_approx_mode = False
    compute_kernel_config.math_fidelity = ttnn.MathFidelity.HiFi4
    compute_kernel_config.unpack_to_dest_mode = [
        ttnn._ttnn.program_descriptor.UnpackToDestMode.UnpackToDestFp32
    ] * 32  #  ttnn.UnpackToDestMode.UnpackToDestFp32

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_source_code,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[[[]]],
        config=compute_kernel_config,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)

    return output


def generate_unary_kernel_from_sfpi_source(sfpi_kernel_name):
    return generate_kernel_from_sfpi_source("unary-sfpi", sfpi_kernel_name)


def main():
    device = ttnn.open_device(device_id=0)
    try:
        input_f32 = torch.tensor([1.0, math.inf, -math.inf, math.nan])

        tt_input_f32 = ttnn.from_torch(input_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_input_bf16 = ttnn.from_torch(input_f32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Expect result should be [1, 1, 1, 0]
        kernel_source_code = generate_unary_kernel_from_sfpi_source("compare")

        # Dump kernel source code to file
        with open(f"compute_unary_equal.cpp", "w") as f:
            f.write(kernel_source_code)

        ttnn_output_f32 = ttnn.zeros_like(tt_input_f32)
        ttnn_output_f32 = generic_unary_kernel(kernel_source_code, tt_input_f32, ttnn_output_f32)

        ttnn_output_bf16 = ttnn.zeros_like(tt_input_bf16)
        ttnn_output_bf16 = generic_unary_kernel(kernel_source_code, tt_input_bf16, ttnn_output_bf16)

        torch_output_f32 = ttnn.to_torch(ttnn_output_f32)
        torch_output_bf16 = ttnn.to_torch(ttnn_output_bf16)

        print(f"input          : {input_f32}")
        print(f"--------------------------------")
        print(f"expected output: {torch.tensor([3.0, 3.0, 3.0, -3.0])}")
        print(f"calculated f32 : {torch_output_f32}")
        print(f"calculated bf16: {torch_output_bf16}")

    finally:
        ttnn.close_device(device)


main()
