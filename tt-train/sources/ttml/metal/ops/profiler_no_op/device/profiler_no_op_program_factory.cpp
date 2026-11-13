// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_no_op_program_factory.hpp"

#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "profiler_no_op_device_operation.hpp"
#include "profiler_no_op_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "reader_profiler_no_op_interleaved_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "writer_profiler_no_op_interleaved_id.cpp";

// Kernel push order in the ProgramDescriptor: reader, then writer (there is no compute kernel).
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0U;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;

}  // namespace

namespace ttml::metal::ops::profiler_no_op::device {

namespace {

// The input's tile rows over the device's compute grid. create_descriptor() pins it against split_work_to_cores, and
// override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& input) {
    const auto& shape = input.logical_shape();
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        shape[0] * shape[1] * ((shape[-2] + tt::constants::TILE_HEIGHT - 1U) / tt::constants::TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor ProfilerNoopProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;

    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor program;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto tensor_shape = input.logical_shape();
    TT_FATAL(tensor_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t Wt = (tensor_shape[-1] + tt::constants::TILE_WIDTH - 1U) /
                  tt::constants::TILE_WIDTH;  // <- number of tiles in inner dimension
    uint32_t Ht = (tensor_shape[-2] + tt::constants::TILE_HEIGHT - 1U) / tt::constants::TILE_HEIGHT;
    uint32_t NC = tensor_shape[0] * tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // compile arguments
    uint32_t block_size = get_block_size(Wt, 4U);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;
    auto data_format = input_data_format;  // tt::DataFormat::Float16_b

    program.cbs.push_back(
        make_cb_descriptor(all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));

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

    {
        std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
        tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
        program.kernels.push_back(  // kReaderKernelIdx
            make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));
    }

    {
        std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
        program.kernels.push_back(  // kWriterKernelIdx
            make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));
    }

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for profiler_no_op
    // -------------------------------------------------------------------------

    // No compute kernels are needed for the profiler_no_op operation

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_rows/start_row follow from the hashed input shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Reader kernel: (input_addr, number_of_rows, offset_in_rows)
        reader.emplace_runtime_args(core, {input_buffer, num_rows, start_row});
        // Writer kernel: (dst_addr, number_of_rows, offset_in_rows)
        writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
    });

    return program;
}

void ProfilerNoopProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;
        reader_runtime_args[core.x][core.y][kInputAddrIdx] = input_addr;
        writer_runtime_args[core.x][core.y][kOutputAddrIdx] = output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(ProfilerNoopOperation, ProfilerNoopProgramFactory);

}  // namespace ttml::metal::ops::profiler_no_op::device
