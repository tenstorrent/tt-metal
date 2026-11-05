// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "silu_bw_device_operation.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/writer_silu_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/reader_silu_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath = "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/compute/silu_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute group 1, then compute
// group 2 when it has cores.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kDLdoutAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kDLdaAddrIdx = 0U;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kDLoutCbIndex = tt::CBIndex::c_1;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_2;
// CBs with intermediate computations
constexpr uint32_t kSigmoidCbIndex = tt::CBIndex::c_3;
constexpr uint32_t kOneMinusSigmoidCbIndex = tt::CBIndex::c_4;
constexpr uint32_t kTimesInputPlusOneCbIndex = tt::CBIndex::c_5;
constexpr uint32_t kTimesSigmoidCbIndex = tt::CBIndex::c_6;

}  // namespace

namespace ttml::metal::ops::silu_bw::device {

namespace {

// The input's tile rows over the device's compute grid. create_descriptor() pins it against split_work_to_cores, and
// override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& input) {
    const auto& shape = input.padded_shape();
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        shape[0] * shape[1] * (shape[-2] / tt::constants::TILE_HEIGHT));
}

}  // namespace

tt::tt_metal::ProgramDescriptor SiLUBackwardProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& dLdout = tensor_args.dL_dout;

    auto* device = input.device();
    tt::tt_metal::ProgramDescriptor program;

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

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Compile arguments
    uint32_t block_size =
        4U;  // We enforce to use block_size of 4. If C % 4 != 0, we will take care of it in the kernels.

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

    // CreateCircularBuffer merged CB core ranges (kernels keep the unmerged set); merge here to match.
    const tt::tt_metal::CoreRangeSet cb_cores = all_cores.merge_ranges();

    program.cbs.push_back(
        make_cb_descriptor(cb_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(
        make_cb_descriptor(cb_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(
        make_cb_descriptor(cb_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(
        make_cb_descriptor(cb_cores, kSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(make_cb_descriptor(
        cb_cores, kOneMinusSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(make_cb_descriptor(
        cb_cores, kTimesInputPlusOneCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));
    program.cbs.push_back(make_cb_descriptor(
        cb_cores, kTimesSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
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

    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    program.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, {}, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_da_buffer).append_to(writer_compile_time_args);
    program.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, {}, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for silu_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    program.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true));

    // Group 2 (if present) compile-time arguments
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
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_rows/start_row follow from the hashed input shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Reader kernel: (input_addr, dLdout_addr, num_rows, offset)
        reader.emplace_runtime_args(core, {input_buffer, dLdout_buffer, num_rows, start_row});
        // Writer kernel: (da_addr, num_rows, offset)
        writer.emplace_runtime_args(core, {dL_da_buffer, num_rows, start_row});
    });

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program descriptor
    // -------------------------------------------------------------------------
    return program;
}

void SiLUBackwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t dLdout_addr = tensor_args.dL_dout.buffer()->address();
    const uint32_t dL_da_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kDLdoutAddrIdx] = dLdout_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kDLdaAddrIdx] = dL_da_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SiLUBackwardDeviceOperation, SiLUBackwardProgramFactory);

}  // namespace ttml::metal::ops::silu_bw::device
