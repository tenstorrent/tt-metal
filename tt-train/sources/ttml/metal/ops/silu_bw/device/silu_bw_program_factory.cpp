// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/writer_silu_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/dataflow/reader_silu_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath = "tt-train/sources/ttml/metal/ops/silu_bw/device/kernels/compute/silu_bw_kernel.cpp";

// Buffer indices
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kDLdoutBufferIdx = 1U;

// Writer buffer indices
constexpr uint32_t kDaBufferIdx = 0;

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

struct SiLUBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SiLUBackwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* dLdout_buffer,
    const tt::tt_metal::Buffer* da_buffer,
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

        // Reader kernel: (input_addr, dLdout_addr, num_rows, offset)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(), dLdout_buffer->address(), num_rows_per_core, num_rows_written});

        // Writer kernel: (da_addr, num_rows, offset)
        SetRuntimeArgs(program, kernels.writer, core, {da_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

SiLUBackwardProgramFactory::cached_program_t SiLUBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& dLdout = tensor_args.dL_dout;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

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
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    uint32_t block_size =
        4U;  // We enforce to use block_size of 4. If C % 4 != 0, we will take care of it in the kernels.

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b

    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_dLdout = create_circular_buffer(
        program, all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_dL_da = create_circular_buffer(
        program, all_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_sigmoid = create_circular_buffer(
        program, all_cores, kSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_one_minus_sigmoid = create_circular_buffer(
        program, all_cores, kOneMinusSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_times_input_plus_one = create_circular_buffer(
        program, all_cores, kTimesInputPlusOneCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_times_sigmoid = create_circular_buffer(
        program, all_cores, kTimesSigmoidCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

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

    SiLUBackwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_da_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for cross_entropy_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            Wt                          // num_inner / TILE_W
        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }
    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        dLdout_buffer,
        dL_da_buffer,
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
        {/* silu_bw_reader_kernel_id  = */ kernels.reader,
         /* silu_bw_writer_kernel_id  = */ kernels.writer,
         /* silu_bw_kernel_group_1_id = */ kernels.compute_group_1,
         /* silu_bw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void SiLUBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& silu_bw_reader_kernel_id = shared_variables.silu_bw_reader_kernel_id;
    auto& silu_bw_writer_kernel_id = shared_variables.silu_bw_writer_kernel_id;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* dLdout_buffer = tensor_args.dL_dout.buffer();

    auto* da_buffer = output.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, silu_bw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, silu_bw_writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kDLdoutBufferIdx] = dLdout_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kDaBufferIdx] = da_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::silu_bw::device
