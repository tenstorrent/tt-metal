// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_program_factory.hpp"

#include <cstdlib>
#include <enchantum/enchantum.hpp>
#include <string>
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

constexpr auto kInputPass1CbIndex = tt::CBIndex::c_0;
constexpr auto kInputPass2CbIndex = tt::CBIndex::c_1;
constexpr auto kInputPass3CbIndex = tt::CBIndex::c_19;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kEpsCbIndex = tt::CBIndex::c_3;
constexpr auto kW0CbIndex = tt::CBIndex::c_4;
constexpr auto kW1CbIndex = tt::CBIndex::c_5;
constexpr auto kW2CbIndex = tt::CBIndex::c_6;
constexpr auto kBiasCbIndex = tt::CBIndex::c_7;
constexpr auto kSumX2CbIndex = tt::CBIndex::c_8;
constexpr auto kSumX4CbIndex = tt::CBIndex::c_9;
constexpr auto kSumX6CbIndex = tt::CBIndex::c_10;
constexpr auto kInvRmsXCbIndex = tt::CBIndex::c_11;
constexpr auto kInvRmsX2CbIndex = tt::CBIndex::c_12;
constexpr auto kInvRmsX3CbIndex = tt::CBIndex::c_13;
constexpr auto kOutputCbIndex = tt::CBIndex::c_14;
constexpr auto kOnesCbIndex = tt::CBIndex::c_15;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_16;
constexpr auto kZeroCbIndex = tt::CBIndex::c_17;
constexpr auto kDebugCbIndex = tt::CBIndex::c_18;

constexpr uint32_t kNumOneTile = 1U;

int get_polynorm_stage_from_env() {
    constexpr int kMinStage = 0;
    constexpr int kMaxStage = 10;
    const char* env = std::getenv("TTML_POLYNORM_FW_STAGE");
    if (env == nullptr) {
        return 0;
    }
    try {
        int stage = std::stoi(env);
        if (stage < kMinStage || stage > kMaxStage) {
            return 0;
        }
        return stage;
    } catch (...) {
        return 0;
    }
}

struct PolyNormForwardKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const PolyNormForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t packed_scaler,
    uint32_t packed_eps,
    uint32_t packed_w0,
    uint32_t packed_w1,
    uint32_t packed_w2,
    uint32_t packed_bias,
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
                num_rows_per_core,
                num_rows_written,
                packed_scaler,
                packed_eps,
                packed_w0,
                packed_w1,
                packed_w2,
                packed_bias,
            });

        SetRuntimeArgs(program, kernels.writer, core, {output_buffer->address(), num_rows_per_core, num_rows_written});
        num_rows_written += num_rows_per_core;
    }
}

}  // namespace

namespace ttml::metal::ops::polynorm_fw::device {

PolyNormForwardProgramFactory::cached_program_t PolyNormForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};
    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "PolyNormForward currently supports BF16 input only");
    const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    const uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    const uint32_t total_rows_to_process = NC * Ht;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const uint32_t block_size = get_block_size(Wt, 4U);
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    [[maybe_unused]] auto cb_input_pass_1 =
        create_circular_buffer(program, all_cores, kInputPass1CbIndex, data_format, tile_size, block_size);
    [[maybe_unused]] auto cb_input_pass_2 =
        create_circular_buffer(program, all_cores, kInputPass2CbIndex, data_format, tile_size, block_size);
    [[maybe_unused]] auto cb_input_pass_3 =
        create_circular_buffer(program, all_cores, kInputPass3CbIndex, data_format, tile_size, block_size);
    [[maybe_unused]] auto cb_scaler =
        create_circular_buffer(program, all_cores, kScalerCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_eps =
        create_circular_buffer(program, all_cores, kEpsCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w0 =
        create_circular_buffer(program, all_cores, kW0CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w1 =
        create_circular_buffer(program, all_cores, kW1CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w2 =
        create_circular_buffer(program, all_cores, kW2CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_bias =
        create_circular_buffer(program, all_cores, kBiasCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x2 =
        create_circular_buffer(program, all_cores, kSumX2CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x4 =
        create_circular_buffer(program, all_cores, kSumX4CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x6 =
        create_circular_buffer(program, all_cores, kSumX6CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x =
        create_circular_buffer(program, all_cores, kInvRmsXCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x2 =
        create_circular_buffer(program, all_cores, kInvRmsX2CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x3 =
        create_circular_buffer(program, all_cores, kInvRmsX3CbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, tile_size, block_size);
    [[maybe_unused]] auto cb_ones =
        create_circular_buffer(program, all_cores, kOnesCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_matmul_reduce =
        create_circular_buffer(program, all_cores, kMatMulReduceCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_zero =
        create_circular_buffer(program, all_cores, kZeroCbIndex, data_format, tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_debug =
        create_circular_buffer(program, all_cores, kDebugCbIndex, data_format, tile_size, 6U);

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

    PolyNormForwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    const uint32_t polynorm_stage = static_cast<uint32_t>(get_polynorm_stage_from_env());
    defines["POLYNORM_STAGE"] = std::to_string(polynorm_stage);
    // Toggle to 1 while debugging fused forward internals.
    constexpr bool kEnablePolyNormDebug = false;
    if constexpr (kEnablePolyNormDebug) {
        defines["POLYNORM_DEBUG"] = "1";
    }

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // Bump this to force fresh JIT binaries when debugging kernel behavior.
    constexpr uint32_t kPolyNormKernelRevision = 13U;
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1, block_size, Wt, polynorm_stage, kPolyNormKernelRevision};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, true);
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2, block_size, Wt, polynorm_stage, kPolyNormKernelRevision};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, true);
    }

    const uint32_t packed_scaler = pack_two_bfloat16_to_uint32(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t packed_eps = pack_two_bfloat16_to_uint32(args.epsilon);
    const uint32_t packed_w0 = pack_two_bfloat16_to_uint32(args.w0);
    const uint32_t packed_w1 = pack_two_bfloat16_to_uint32(args.w1);
    const uint32_t packed_w2 = pack_two_bfloat16_to_uint32(args.w2);
    const uint32_t packed_bias = pack_two_bfloat16_to_uint32(args.bias);

    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        output_buffer,
        packed_scaler,
        packed_eps,
        packed_w0,
        packed_w1,
        packed_w2,
        packed_bias,
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

void PolyNormForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* input_buffer = tensor_args.input.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    const uint32_t packed_scaler =
        pack_two_bfloat16_to_uint32(1.0F / static_cast<float>(tensor_args.input.logical_shape()[-1]));
    const uint32_t packed_eps = pack_two_bfloat16_to_uint32(operation_attributes.epsilon);
    const uint32_t packed_w0 = pack_two_bfloat16_to_uint32(operation_attributes.w0);
    const uint32_t packed_w1 = pack_two_bfloat16_to_uint32(operation_attributes.w1);
    const uint32_t packed_w2 = pack_two_bfloat16_to_uint32(operation_attributes.w2);
    const uint32_t packed_bias = pack_two_bfloat16_to_uint32(operation_attributes.bias);

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};

        auto& rr = reader_runtime_args[core.x][core.y];
        rr[0] = input_buffer->address();
        rr[3] = packed_scaler;
        rr[4] = packed_eps;
        rr[5] = packed_w0;
        rr[6] = packed_w1;
        rr[7] = packed_w2;
        rr[8] = packed_bias;

        auto& wr = writer_runtime_args[core.x][core.y];
        wr[kWriterOutputBufferIdx] = output_buffer->address();
    }
}

}  // namespace ttml::metal::ops::polynorm_fw::device
