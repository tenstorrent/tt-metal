// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/writer_silu_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/reader_silu_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath = "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/compute/silu_bw_kernel.cpp";

// CBs with input data
constexpr uint32_t kInputCbIndex = tt::CBIndex::c_0;
constexpr uint32_t kDLoutCbIndex = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t kDLdaCbIndex = tt::CBIndex::c_2;
// CBs with intermediate computations
constexpr uint32_t kSigmoidCbIndex = tt::CBIndex::c_3;
constexpr uint32_t kOneMinusSigmoidCbIndex = tt::CBIndex::c_4;
constexpr uint32_t kTimesInputPlusOneCbIndex = tt::CBIndex::c_5;
constexpr uint32_t kTimesSigmoidCbIndex = tt::CBIndex::c_6;

}  // namespace

namespace ttml::metal::ops::silu_bw::device {

tt::tt_metal::ProgramDescriptor SiLUBackwardProgramFactory::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Data formats, tile sizes and the work split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& dLdout = tensor_args.dL_dout;
    auto* device = input.device();

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // We enforce a block_size of 4. If C % 4 != 0, the kernels take care of the remainder.
    uint32_t block_size = 4U;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* dLdout_buffer = dLdout.buffer();
    TT_FATAL(
        dLdout_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dLdout_buffer->buffer_type()));

    auto* dL_da_buffer = output.buffer();
    TT_FATAL(
        dL_da_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_da buffer must be in DRAM. dL_da buffer of type {}",
        enchantum::to_string(dL_da_buffer->buffer_type()));

    // -------------------------------------------------------------------------
    // 2) Circular buffers
    // -------------------------------------------------------------------------
    tt::tt_metal::ProgramDescriptor program;

    const uint32_t twice_block_size = 2U * block_size;
    for (uint32_t cb_index :
         {kInputCbIndex,
          kDLoutCbIndex,
          kDLdaCbIndex,
          kSigmoidCbIndex,
          kOneMinusSigmoidCbIndex,
          kTimesInputPlusOneCbIndex,
          kTimesSigmoidCbIndex}) {
        program.cbs.push_back(make_cb_descriptor(
            all_cores, cb_index, input_data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    }

    // -------------------------------------------------------------------------
    // 3) Reader/writer kernels
    // -------------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    auto reader = make_reader_kernel_descriptor(all_cores, reader_compile_time_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_da_buffer).append_to(writer_compile_time_args);
    auto writer = make_writer_kernel_descriptor(all_cores, writer_compile_time_args, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Compute kernels: one per core group, the row count is a compile-time argument
    // -------------------------------------------------------------------------
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };
    program.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            Wt                          // num_inner / TILE_W
        };
        program.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    // -------------------------------------------------------------------------
    // 5) Per-core runtime args. Buffers are bound, not addressed: the framework fills in the address when the
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
            // Reader kernel: (input_addr, dLdout_addr, num_rows, offset)
            reader.emplace_runtime_args(core, {input_buffer, dLdout_buffer, num_rows, start_row});
            // Writer kernel: (da_addr, num_rows, offset)
            writer.emplace_runtime_args(core, {dL_da_buffer, num_rows, start_row});
        });

    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    return program;
}

}  // namespace ttml::metal::ops::silu_bw::device
