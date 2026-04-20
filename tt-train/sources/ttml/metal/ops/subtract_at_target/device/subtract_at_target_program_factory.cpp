// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_program_factory.hpp"

#include <cstring>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "subtract_at_target_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_reader.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/subtract_at_target/device/kernels/dataflow/subtract_at_target_writer.cpp";

// Reader runtime arg indices
constexpr uint32_t kInputBufferIdx = 0U;
constexpr uint32_t kTargetBufferIdx = 1U;
constexpr uint32_t kFirstVIdx = 4U;
constexpr uint32_t kLastVIdx = 5U;
constexpr uint32_t kSubtractValueIdx = 6U;

// Writer runtime arg indices
constexpr uint32_t kOutputBufferIdx = 0U;

// CB indices
constexpr auto kTargetCbIndex = tt::CBIndex::c_0;        // scratch: target page (uint32)
constexpr auto kInputScratchCbIndex = tt::CBIndex::c_1;  // scratch: one input tile (bfloat16)
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;        // output tiles (bfloat16)

constexpr uint32_t kPageElementsNumber = 32U;
constexpr uint32_t kNumOutputTiles = 2U;  // double-buffered

}  // namespace

namespace ttml::metal::ops::subtract_at_target::device {

SubtractAtTargetProgramFactory::cached_program_t SubtractAtTargetProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    TT_FATAL(
        datatype_to_dataformat_converter(input.dtype()) == tt::DataFormat::Float16_b,
        "subtract_at_target: input must be BFLOAT16");

    const uint32_t bfloat16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    const auto padded_shape = input.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "subtract_at_target: input must be rank 4");

    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    const uint32_t total_rows = NC * Ht;

    const uint32_t target_page_size = target.logical_shape()[-1] * target.element_size();
    const uint32_t target_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * kPageElementsNumber;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------

    create_circular_buffer(
        program, all_cores, kTargetCbIndex, tt::DataFormat::UInt32, target_read_page_size, /*num_tiles=*/1U);

    create_circular_buffer(
        program, all_cores, kInputScratchCbIndex, tt::DataFormat::Float16_b, bfloat16_tile_bytes, /*num_tiles=*/1U);

    create_circular_buffer(
        program, all_cores, kOutputCbIndex, tt::DataFormat::Float16_b, bfloat16_tile_bytes, kNumOutputTiles);

    // -------------------------------------------------------------------------
    // Reader kernel
    // -------------------------------------------------------------------------

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: input buffer must be DRAM, got {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* target_buffer = target.buffer();
    TT_FATAL(
        target_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: target buffer must be DRAM, got {}",
        enchantum::to_string(target_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "subtract_at_target: output buffer must be DRAM, got {}",
        enchantum::to_string(output_buffer->buffer_type()));

    std::vector<uint32_t> reader_ct_args{Wt, Ht, target_page_size, target_read_page_size};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(target_buffer).append_to(reader_ct_args);

    auto reader_kernel = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    // -------------------------------------------------------------------------
    // Writer kernel
    // -------------------------------------------------------------------------

    std::vector<uint32_t> writer_ct_args{Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);

    auto writer_kernel = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // Per-core runtime args
    // -------------------------------------------------------------------------

    uint32_t subtract_bits;
    std::memcpy(&subtract_bits, &operation_attributes.subtract_value, sizeof(uint32_t));

    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const uint32_t num_rows_this_core = core_group_1.contains(core)   ? num_rows_per_core_group_1
                                            : core_group_2.contains(core) ? num_rows_per_core_group_2
                                                                          : 0U;
        TT_FATAL(num_rows_this_core > 0U, "subtract_at_target: core not in any group");

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {input_buffer->address(),
             target_buffer->address(),
             num_rows_this_core,
             num_rows_written,
             operation_attributes.first_v,
             operation_attributes.last_v,
             subtract_bits});

        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), num_rows_this_core, num_rows_written});

        num_rows_written += num_rows_this_core;
    }

    return cached_program_t{
        std::move(program), {reader_kernel, writer_kernel, core_group_1, core_group_2, num_cores, num_cores_y}};
}

void SubtractAtTargetProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto* input_buffer = tensor_args.input.buffer();
    auto* target_buffer = tensor_args.target.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    auto& reader_args = GetRuntimeArgs(program, sv.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, sv.writer_kernel_id);

    for (uint32_t i = 0; i < sv.num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / sv.num_cores_y, i % sv.num_cores_y};

        {
            auto& args = reader_args[core.x][core.y];
            args[kInputBufferIdx] = input_buffer->address();
            args[kTargetBufferIdx] = target_buffer->address();
            args[kFirstVIdx] = operation_attributes.first_v;
            args[kLastVIdx] = operation_attributes.last_v;
            uint32_t subtract_bits;
            std::memcpy(&subtract_bits, &operation_attributes.subtract_value, sizeof(uint32_t));
            args[kSubtractValueIdx] = subtract_bits;
        }
        {
            auto& args = writer_args[core.x][core.y];
            args[kOutputBufferIdx] = output_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::subtract_at_target::device
