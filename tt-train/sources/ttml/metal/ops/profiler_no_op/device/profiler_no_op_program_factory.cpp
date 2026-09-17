// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_no_op_program_factory.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "reader_profiler_no_op_interleaved_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/profiler_no_op/device/kernels/dataflow/"
    "writer_profiler_no_op_interleaved_id.cpp";

constexpr auto kInputCbIndex = tt::CBIndex::c_0;

}  // namespace

namespace ttml::metal::ops::profiler_no_op::device {

tt::tt_metal::ProgramDescriptor ProfilerNoopProgramFactory::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Data formats, tile sizes and the work split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    auto* device = input.device();

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

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t block_size = get_block_size(Wt, 4U);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

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

    // -------------------------------------------------------------------------
    // 2) Circular buffers
    // -------------------------------------------------------------------------
    tt::tt_metal::ProgramDescriptor program;

    const uint32_t twice_block_size = 2U * block_size;
    program.cbs.push_back(make_cb_descriptor(
        all_cores, kInputCbIndex, input_data_format, bfloat16_single_tile_size_bytes, twice_block_size));

    // -------------------------------------------------------------------------
    // 3) Reader/writer kernels (no compute kernel: the op only moves data)
    // -------------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    auto reader = make_reader_kernel_descriptor(all_cores, reader_compile_time_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    auto writer = make_writer_kernel_descriptor(all_cores, writer_compile_time_args, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Per-core runtime args. Buffers are bound, not addressed: the framework fills in the address when the
    //    program is built and again on every cache hit.
    // -------------------------------------------------------------------------
    for_each_core_with_work(
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        [&](const CoreWork& work) {
            const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
            // Reader kernel: (input_addr, number_of_rows, offset_in_rows)
            reader.emplace_runtime_args(core, {input_buffer, num_rows, start_row});
            // Writer kernel: (dst_addr, number_of_rows, offset_in_rows)
            writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
        });

    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    return program;
}

}  // namespace ttml::metal::ops::profiler_no_op::device
