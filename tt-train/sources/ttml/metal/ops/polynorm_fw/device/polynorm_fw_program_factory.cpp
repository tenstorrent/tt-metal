// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_program_factory.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/dataflow/"
    "reader_polynorm_fw_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/dataflow/"
    "writer_polynorm_fw_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/compute/polynorm_fw_kernel.cpp";

constexpr uint32_t kWriterOutputBufferIdx = 0U;

// CBs with input data / scalar parameters
constexpr auto kInputPass1CbIndex = tt::CBIndex::c_0;
constexpr auto kInputPass2CbIndex = tt::CBIndex::c_1;
constexpr auto kInputPass3CbIndex = tt::CBIndex::c_2;
constexpr auto kScalerCbIndex = tt::CBIndex::c_3;
constexpr auto kEpsCbIndex = tt::CBIndex::c_4;
constexpr auto kW0CbIndex = tt::CBIndex::c_5;
constexpr auto kW1CbIndex = tt::CBIndex::c_6;
constexpr auto kW2CbIndex = tt::CBIndex::c_7;
constexpr auto kBiasCbIndex = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr auto kSumX2CbIndex = tt::CBIndex::c_9;
constexpr auto kSumX4CbIndex = tt::CBIndex::c_10;
constexpr auto kSumX6CbIndex = tt::CBIndex::c_11;
constexpr auto kInvRmsXCbIndex = tt::CBIndex::c_12;
constexpr auto kInvRmsX2CbIndex = tt::CBIndex::c_13;
constexpr auto kInvRmsX3CbIndex = tt::CBIndex::c_14;
// CBs with output data
constexpr auto kOutputCbIndex = tt::CBIndex::c_15;

constexpr uint32_t kNumOneTile = 1U;

struct PolyNormForwardKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

// Assign reader/writer runtime arguments for every active core.
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const PolyNormForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* weight_buffer,
    const tt::tt_metal::Buffer* bias_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t scaler_fp32_bits,
    uint32_t eps_fp32_bits,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core = 0U;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {
                input_buffer->address(),
                weight_buffer->address(),
                bias_buffer->address(),
                num_rows_per_core,
                num_rows_written,
                scaler_fp32_bits,
                eps_fp32_bits,
            });

        SetRuntimeArgs(program, kernels.writer, core, {output_buffer->address(), num_rows_per_core, num_rows_written});
        num_rows_written += num_rows_per_core;
    }
}

}  // namespace

namespace ttml::metal::ops::polynorm_fw::device {

// Build and cache the full PolyNorm forward program (reader/compute/writer kernels + CB layout).
PolyNormForwardProgramFactory::cached_program_t PolyNormForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    auto* device = input.device();

    tt::tt_metal::Program program{};
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "PolyNormForward currently supports BF16 input only");
    const uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    const auto padded_tensor_shape = input.padded_shape();
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    const uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    const uint32_t total_rows_to_process = NC * Ht;

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    constexpr uint32_t block_size = 4U;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    [[maybe_unused]] auto cb_input_pass_1 =
        create_circular_buffer(program, all_cores, kInputPass1CbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_input_pass_2 =
        create_circular_buffer(program, all_cores, kInputPass2CbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_input_pass_3 =
        create_circular_buffer(program, all_cores, kInputPass3CbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_eps = create_circular_buffer(
        program, all_cores, kEpsCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w0 =
        create_circular_buffer(program, all_cores, kW0CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w1 =
        create_circular_buffer(program, all_cores, kW1CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w2 =
        create_circular_buffer(program, all_cores, kW2CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_bias =
        create_circular_buffer(program, all_cores, kBiasCbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x2 = create_circular_buffer(
        program, all_cores, kSumX2CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x4 = create_circular_buffer(
        program, all_cores, kSumX4CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x6 = create_circular_buffer(
        program, all_cores, kSumX6CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x =
        create_circular_buffer(program, all_cores, kInvRmsXCbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x2 =
        create_circular_buffer(program, all_cores, kInvRmsX2CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x3 =
        create_circular_buffer(program, all_cores, kInvRmsX3CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_tile_size, block_size);
    auto* input_buffer = input.buffer();
    auto* weight_buffer = weight.buffer();
    auto* bias_buffer = bias.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));
    TT_FATAL(
        weight_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "weight buffer must be in DRAM. weight buffer of type {}",
        enchantum::to_string(weight_buffer->buffer_type()));
    TT_FATAL(
        bias_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "bias buffer must be in DRAM. bias buffer of type {}",
        enchantum::to_string(bias_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    PolyNormForwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    std::vector<uint32_t> compute_group_1_args = {num_rows_per_core_group_1, block_size, Wt};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, true);
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_rows_per_core_group_2, block_size, Wt};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, true);
    }

    const uint32_t scaler_fp32_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(args.epsilon);

    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        weight_buffer,
        bias_buffer,
        output_buffer,
        scaler_fp32_bits,
        eps_fp32_bits,
        num_cores,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2);

    return cached_program_t{
        std::move(program),
        {
            kernels.reader,
            kernels.writer,
            kernels.compute_group_1,
            kernels.compute_group_2,
            core_group_1,
            core_group_2,
            num_cores,
            num_cores_y,
        }};
}

// Update runtime addresses/scalars when operation attributes or buffers change.
void PolyNormForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* input_buffer = tensor_args.input.buffer();
    auto* weight_buffer = tensor_args.weight.buffer();
    auto* bias_buffer = tensor_args.bias.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    const uint32_t scaler_fp32_bits =
        std::bit_cast<uint32_t>(1.0F / static_cast<float>(tensor_args.input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};

        auto& rr = reader_runtime_args[core.x][core.y];
        rr[0] = input_buffer->address();
        rr[1] = weight_buffer->address();
        rr[2] = bias_buffer->address();
        rr[5] = scaler_fp32_bits;
        rr[6] = eps_fp32_bits;

        auto& wr = writer_runtime_args[core.x][core.y];
        wr[kWriterOutputBufferIdx] = output_buffer->address();
    }
}

}  // namespace ttml::metal::ops::polynorm_fw::device
