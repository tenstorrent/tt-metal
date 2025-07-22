// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_no_op_program_factory.hpp"
#include <enchantum/enchantum.hpp>

#include <core/ttnn_all_includes.hpp>

#include "metal/ops/common/program_utils.hpp"
#include "profiler_no_op_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "reader_profiler_no_op_interleaved_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "writer_profiler_no_op_interleaved_id.cpp";

// reader runtime args
constexpr uint32_t kInputBufferIdx = 0;
// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;

}  // namespace

namespace ttml::metal::ops::profiler_no_op::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct CrossEntropyBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const CrossEntropyBackwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many rows this core will process
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        // Reader kernel: (input_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(program, kernels.reader, core, {input_buffer->address(), num_rows_per_core, num_rows_written});

        // Writer kernel: (dst_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(program, kernels.writer, core, {output_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

ProfilerNoopProgramFactory::cached_program_t ProfilerNoopProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;

    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;  // <- number of tiles in inner dimension
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // compile arguments
    uint32_t block_size = get_block_size(Wt, 4U);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;
    auto data_format = input_data_format;  // tt::DataFormat::Float16_b

    auto cb_dataflow = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;

    CrossEntropyBackwardKernels kernels;
    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */
        {block_size, Wt},
        defines,
        kReaderKernelPath);

    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ {block_size, Wt}, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for profiler_no_op
    // -------------------------------------------------------------------------

    // No compute kernels are needed for the profiler_no_op operation

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------

    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        output_buffer,
        num_cores,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------

    return cached_program_t{
        std::move(program),
        {/* profiler_no_opreader_kernel_id  = */ kernels.reader,
         /* profiler_no_op_writer_kernel_id  = */ kernels.writer,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void ProfilerNoopProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& profiler_no_op_reader_kernel_id = shared_variables.reader_kernel_id;
    auto& profiler_no_op_writer_kernel_id = shared_variables.writer_kernel_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, profiler_no_op_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, profiler_no_op_writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::profiler_no_op::device
