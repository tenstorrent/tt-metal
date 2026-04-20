// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_program_factory.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/reader_polynorm_bw_interleaved_start_id.cpp";
constexpr auto kDLdxWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/writer_polynorm_bw_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/compute/polynorm_bw_kernel.cpp";

constexpr uint32_t kWriterOutputBufferIdx = 0U;

// Data input CBs (shared across all sequential passes)
constexpr auto kXCbIndex = tt::CBIndex::c_0;
constexpr auto kDoutCbIndex = tt::CBIndex::c_1;
constexpr auto kDbAccCbIndex = tt::CBIndex::c_2;

// Scalar/constant CBs
constexpr auto kScalerCbIndex = tt::CBIndex::c_3;
constexpr auto kEpsCbIndex = tt::CBIndex::c_4;
constexpr auto kOneCbIndex = tt::CBIndex::c_5;
constexpr auto kW0CbIndex = tt::CBIndex::c_6;
constexpr auto kW1CbIndex = tt::CBIndex::c_7;
constexpr auto kW2CbIndex = tt::CBIndex::c_8;
// Reader-only scratch CB used to read weight scalars before publishing constants.
constexpr auto kWeightScalarScratchCbIndex = tt::CBIndex::c_23;

// Intermediate per-row sum tiles (Float32)
constexpr auto kSumX2CbIndex = tt::CBIndex::c_9;
constexpr auto kSumX4CbIndex = tt::CBIndex::c_10;
constexpr auto kSumX6CbIndex = tt::CBIndex::c_11;
constexpr auto kSumXdoutCbIndex = tt::CBIndex::c_12;
constexpr auto kSumX2doutCbIndex = tt::CBIndex::c_13;
constexpr auto kSumX3doutCbIndex = tt::CBIndex::c_14;

// Scalar results (bfloat16)
constexpr auto kInvRmsXCbIndex = tt::CBIndex::c_15;
constexpr auto kInvRmsX2CbIndex = tt::CBIndex::c_16;
constexpr auto kInvRmsX3CbIndex = tt::CBIndex::c_17;

// Coeff tiles (bfloat16)
constexpr auto kCoeff1CbIndex = tt::CBIndex::c_18;
constexpr auto kCoeff2CbIndex = tt::CBIndex::c_19;
constexpr auto kCoeff3CbIndex = tt::CBIndex::c_20;

// Output CBs
constexpr auto kOutputCbIndex = tt::CBIndex::c_21;
constexpr auto kPackedPartialsOutputCbIndex = tt::CBIndex::c_22;

constexpr uint32_t kNumOneTile = 1U;

struct PolyNormBackwardKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle dL_dx_writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const PolyNormBackwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* dL_dout_buffer,
    const tt::tt_metal::Buffer* weight_buffer,
    const tt::tt_metal::Buffer* dL_dx_output_buffer,
    const tt::tt_metal::Buffer* packed_partials_output_buffer,
    const uint32_t scaler_fp32_bits,
    const uint32_t eps_fp32_bits,
    const uint32_t num_inner,
    const uint32_t num_cores,
    const uint32_t num_cores_y,
    const uint32_t num_rows_per_core_group_1,
    const uint32_t num_rows_per_core_group_2,
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
                dL_dout_buffer->address(),
                weight_buffer->address(),
                num_rows_per_core,
                num_rows_written,
                scaler_fp32_bits,
                eps_fp32_bits,
            });

        SetRuntimeArgs(
            program,
            kernels.dL_dx_writer,
            core,
            {dL_dx_output_buffer->address(),
             packed_partials_output_buffer->address(),
             num_rows_per_core,
             num_rows_written});

        auto compute_kernel = core_group_1.contains(core) ? kernels.compute_group_1 : kernels.compute_group_2;
        SetRuntimeArgs(program, compute_kernel, core, {num_inner});
        num_rows_written += num_rows_per_core;
    }
}

}  // namespace

namespace ttml::metal::ops::polynorm3_bw::device {

PolyNorm3BackwardProgramFactory::cached_program_t PolyNorm3BackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& dL_dout = tensor_args.dL_dout;
    auto* device = input.device();

    tt::tt_metal::Program program{};
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "PolyNormBackward currently supports BF16 input only");
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

    // Data input CBs (shared across all passes)
    [[maybe_unused]] auto cb_x =
        create_circular_buffer(program, all_cores, kXCbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_dout =
        create_circular_buffer(program, all_cores, kDoutCbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_db_acc = create_circular_buffer(
        program, all_cores, kDbAccCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);

    // Scalar/constant CBs
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_eps = create_circular_buffer(
        program, all_cores, kEpsCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_one = create_circular_buffer(
        program, all_cores, kOneCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w0 =
        create_circular_buffer(program, all_cores, kW0CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w1 =
        create_circular_buffer(program, all_cores, kW1CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_w2 =
        create_circular_buffer(program, all_cores, kW2CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_weight_scalar_scratch = create_circular_buffer(
        program, all_cores, kWeightScalarScratchCbIndex, data_format, bfloat16_tile_size, kNumOneTile);

    // Intermediate per-row sum tiles (Float32)
    [[maybe_unused]] auto cb_sum_x2 = create_circular_buffer(
        program, all_cores, kSumX2CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x4 = create_circular_buffer(
        program, all_cores, kSumX4CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x6 = create_circular_buffer(
        program, all_cores, kSumX6CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_xdout = create_circular_buffer(
        program, all_cores, kSumXdoutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x2dout = create_circular_buffer(
        program, all_cores, kSumX2doutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_sum_x3dout = create_circular_buffer(
        program, all_cores, kSumX3doutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile);

    // Scalar results (bfloat16)
    [[maybe_unused]] auto cb_inv_rms_x =
        create_circular_buffer(program, all_cores, kInvRmsXCbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x2 =
        create_circular_buffer(program, all_cores, kInvRmsX2CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_inv_rms_x3 =
        create_circular_buffer(program, all_cores, kInvRmsX3CbIndex, data_format, bfloat16_tile_size, kNumOneTile);

    // Coeff tiles (bfloat16)
    [[maybe_unused]] auto cb_coeff_1 =
        create_circular_buffer(program, all_cores, kCoeff1CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_coeff_2 =
        create_circular_buffer(program, all_cores, kCoeff2CbIndex, data_format, bfloat16_tile_size, kNumOneTile);
    [[maybe_unused]] auto cb_coeff_3 =
        create_circular_buffer(program, all_cores, kCoeff3CbIndex, data_format, bfloat16_tile_size, kNumOneTile);

    // Output CBs
    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_tile_size, block_size);
    [[maybe_unused]] auto cb_packed_partials_output = create_circular_buffer(
        program, all_cores, kPackedPartialsOutputCbIndex, tt::DataFormat::Float32, float32_tile_size, block_size);

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* dL_dout_buffer = dL_dout.buffer();
    TT_FATAL(
        dL_dout_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dL_dout_buffer->buffer_type()));

    const auto& weight = tensor_args.weight;
    auto* weight_buffer = weight.buffer();
    TT_FATAL(
        weight_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Weight buffer must be in DRAM. Weight buffer of type {}",
        enchantum::to_string(weight_buffer->buffer_type()));

    auto* dL_dx_output_buffer = output[0].buffer();
    TT_FATAL(
        dL_dx_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "dL_dx output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(dL_dx_output_buffer->buffer_type()));
    auto* packed_partials_output_buffer = output[1].buffer();
    TT_FATAL(
        packed_partials_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Packed partials output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(packed_partials_output_buffer->buffer_type()));

    PolyNormBackwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dL_dout_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    constexpr uint32_t packed_partials_wt = 4U;
    std::vector<uint32_t> dL_dx_writer_compile_time_args{block_size, Wt, packed_partials_wt};
    tt::tt_metal::TensorAccessorArgs(dL_dx_output_buffer).append_to(dL_dx_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(packed_partials_output_buffer).append_to(dL_dx_writer_compile_time_args);
    kernels.dL_dx_writer =
        create_writer_kernel(program, all_cores, dL_dx_writer_compile_time_args, defines, kDLdxWriterKernelPath);

    std::vector<uint32_t> compute_group_1_args = {num_rows_per_core_group_1, block_size};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, true);
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_rows_per_core_group_2, block_size};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, true);
    }

    const uint32_t scaler_fp32_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(args.epsilon);

    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        dL_dout_buffer,
        weight_buffer,
        dL_dx_output_buffer,
        packed_partials_output_buffer,
        scaler_fp32_bits,
        eps_fp32_bits,
        Wt,
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
            kernels.dL_dx_writer,
            kernels.compute_group_1,
            kernels.compute_group_2,
            core_group_1,
            core_group_2,
            num_cores,
            num_cores_y,
        }};
}

void PolyNorm3BackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* input_buffer = tensor_args.input.buffer();
    auto* dL_dout_buffer = tensor_args.dL_dout.buffer();
    auto* weight_buffer = tensor_args.weight.buffer();
    auto* dL_dx_output_buffer = tensor_return_value[0].buffer();
    auto* packed_partials_output_buffer = tensor_return_value[1].buffer();

    const uint32_t scaler_fp32_bits =
        std::bit_cast<uint32_t>(1.0F / static_cast<float>(tensor_args.input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& dL_dx_writer_runtime_args = GetRuntimeArgs(program, shared.dL_dx_writer_kernel_id);

    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};

        auto& rr = reader_runtime_args[core.x][core.y];
        rr[0] = input_buffer->address();
        rr[1] = dL_dout_buffer->address();
        rr[2] = weight_buffer->address();
        rr[5] = scaler_fp32_bits;
        rr[6] = eps_fp32_bits;

        auto& dL_dx_wr = dL_dx_writer_runtime_args[core.x][core.y];
        dL_dx_wr[kWriterOutputBufferIdx] = dL_dx_output_buffer->address();
        dL_dx_wr[1] = packed_partials_output_buffer->address();
    }
}

}  // namespace ttml::metal::ops::polynorm3_bw::device
