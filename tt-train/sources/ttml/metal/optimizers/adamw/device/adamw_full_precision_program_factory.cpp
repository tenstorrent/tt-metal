// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_full_precision_program_factory.hpp"

#include <common/TracyQueue.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "adamw_device_operation_types.hpp"
#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw/device/kernels/dataflow/"
    "reader_adamw.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw/device/kernels/dataflow/"
    "writer_adamw.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw/device/kernels/compute/adamw_kernel.cpp";

// reader runtime args
constexpr uint32_t kParamAddrIdx = 0;
constexpr uint32_t kGradAddrIdx = 1U;
constexpr uint32_t kExpAvgAddrIdx = 2U;
constexpr uint32_t kExpAvgSqAddrIdx = 3U;
constexpr uint32_t kMaxExpAvgSqAddrIdx = 4U;
// compute runtime args
constexpr uint32_t kComputeBeta1Idx = 0U;
constexpr uint32_t kComputeBeta2Idx = 1U;
constexpr uint32_t kComputeEpsilonIdx = 2U;
constexpr uint32_t kComputeStepSizeIdx = 3U;
constexpr uint32_t kComputeInvSqrtBiasCorrection2Idx = 4U;
constexpr uint32_t kComputeOneMinusBeta1Idx = 5U;
constexpr uint32_t kComputeOneMinusBeta2Idx = 6U;
constexpr uint32_t kComputeDecayFactorIdx = 7U;
constexpr uint32_t kComputeSeedIdx = 8U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0;
constexpr uint32_t kExpAvgAddrIdxOut = 1U;
constexpr uint32_t kExpAvgSqAddrIdxOut = 2U;
constexpr uint32_t kMaxExpAvgSqAddrIdxOut = 3U;

constexpr auto kParamCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kExpAvgCbIndex = tt::CBIndex::c_2;
constexpr auto kExpAvgSqCbIndex = tt::CBIndex::c_3;
constexpr auto kMaxExpAvgSqInCbIndex = tt::CBIndex::c_4;

constexpr auto kOutputCbIndex = tt::CBIndex::c_16;
constexpr auto kExpAvgOutCbIndex = tt::CBIndex::c_17;
constexpr auto kExpAvgSqOutCbIndex = tt::CBIndex::c_18;
constexpr auto kMaxExpAvgSqOutCbIndex = tt::CBIndex::c_19;

constexpr auto kMomentumCbIndex = tt::CBIndex::c_24;
constexpr auto kVarianceCbIndex = tt::CBIndex::c_25;
constexpr auto kMaxExpAvgSqCbIndex = tt::CBIndex::c_26;

}  // namespace

namespace ttml::metal::optimizers::adamw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct AdamWFullPrecisionKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args_full_precision(
    tt::tt_metal::Program& program,
    const AdamWFullPrecisionKernels& kernels,
    const tt::tt_metal::Buffer* param_buffer,
    const tt::tt_metal::Buffer* grad_buffer,
    const tt::tt_metal::Buffer* exp_avg_buffer,
    const tt::tt_metal::Buffer* exp_avg_sq_buffer,
    const tt::tt_metal::Buffer* max_exp_avg_sq_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_pow,
    const float beta2_pow,
    const float epsilon,
    const float weight_decay,
    [[maybe_unused]] const uint32_t step,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    float bias_correction1 = 1.0f - beta1_pow;
    float bias_correction2 = 1.0f - beta2_pow;
    float step_size = lr / bias_correction1;
    float inv_sqrt_bc2 = 1.0f / std::sqrt(bias_correction2);
    float decay_factor = 1.0f - lr * weight_decay;

    // Update:
    // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))

    // Hyperparameters are common for all cores
    std::vector<uint32_t> compute_common_args = {
        std::bit_cast<uint32_t>(beta1),
        std::bit_cast<uint32_t>(beta2),
        std::bit_cast<uint32_t>(epsilon),
        std::bit_cast<uint32_t>(step_size),
        std::bit_cast<uint32_t>(inv_sqrt_bc2),
        std::bit_cast<uint32_t>(one_minus_beta1),
        std::bit_cast<uint32_t>(one_minus_beta2),
        std::bit_cast<uint32_t>(decay_factor),
        0U};  // seed=0, stochastic rounding disabled
    tt::tt_metal::SetCommonRuntimeArgs(program, kernels.compute_group_1, compute_common_args);
    if (!core_group_2.ranges().empty()) {
        tt::tt_metal::SetCommonRuntimeArgs(program, kernels.compute_group_2, compute_common_args);
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many tiles this core will process
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core {} not in specified core ranges", core);
        }

        // Reader kernel
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {param_buffer->address(),
             grad_buffer->address(),
             exp_avg_buffer->address(),
             exp_avg_sq_buffer->address(),
             max_exp_avg_sq_buffer != nullptr ? max_exp_avg_sq_buffer->address() : 0U,
             num_tiles_per_core,
             num_tiles_written});

        // Writer kernel
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(),
             exp_avg_buffer->address(),
             exp_avg_sq_buffer->address(),
             max_exp_avg_sq_buffer != nullptr ? max_exp_avg_sq_buffer->address() : 0U,
             num_tiles_per_core,
             num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }
}

AdamWFullPrecisionProgramFactory::cached_program_t AdamWFullPrecisionProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& param = tensor_args.param;
    const auto& grad = tensor_args.grad;
    const auto& exp_avg = tensor_args.exp_avg;
    const auto& exp_avg_sq = tensor_args.exp_avg_sq;
    const auto& max_exp_avg_sq_opt = tensor_args.max_exp_avg_sq;
    const auto& lr = operation_attributes.lr;
    const auto& beta1 = operation_attributes.beta1;
    const auto& beta2 = operation_attributes.beta2;
    const auto& beta1_pow = operation_attributes.beta1_pow;
    const auto& beta2_pow = operation_attributes.beta2_pow;
    const auto& epsilon = operation_attributes.epsilon;
    const auto& weight_decay = operation_attributes.weight_decay;
    const auto& amsgrad = operation_attributes.amsgrad;
    const auto& step = operation_attributes.step;

    auto* device = param.device();

    tt::tt_metal::Program program{};

    tt::DataFormat fp32_data_format = tt::DataFormat::Float32;
    tt::DataFormat bf16_data_format = tt::DataFormat::Float16_b;

    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);
    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = param.padded_shape();
    auto padded_tensor_volume = param.physical_volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded param tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t total_tiles_to_process = padded_tensor_volume / tt::constants::TILE_HW;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles_to_process);

    uint32_t block_size = std::min(2U, num_tiles_per_core_group_1);
    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t num_input_tiles = twice_block_size;
    const uint32_t num_output_tiles = twice_block_size;

    // fp32 param (master weights)
    [[maybe_unused]] auto cb_param = create_circular_buffer(
        program, all_cores, kParamCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_input_tiles);

    // bf16 grad
    [[maybe_unused]] auto cb_grad = create_circular_buffer(
        program, all_cores, kGradCbIndex, bf16_data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    // fp32 momentum buffers
    [[maybe_unused]] auto cb_exp_avg = create_circular_buffer(
        program, all_cores, kExpAvgCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_exp_avg_sq = create_circular_buffer(
        program, all_cores, kExpAvgSqCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_input_tiles);

    // fp32 output (updated master weights)
    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_exp_avg_out = create_circular_buffer(
        program, all_cores, kExpAvgOutCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_exp_avg_sq_out = create_circular_buffer(
        program, all_cores, kExpAvgSqOutCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_max_exp_avg_sq_in = create_circular_buffer(
        program, all_cores, kMaxExpAvgSqInCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_max_exp_avg_sq_out = create_circular_buffer(
        program, all_cores, kMaxExpAvgSqOutCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_max_exp_avg_sq = create_circular_buffer(
        program, all_cores, kMaxExpAvgSqCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_momentum = create_circular_buffer(
        program, all_cores, kMomentumCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_variance = create_circular_buffer(
        program, all_cores, kVarianceCbIndex, fp32_data_format, float32_single_tile_size_bytes, num_output_tiles);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* param_buffer = param.buffer();
    auto* grad_buffer = grad.buffer();
    auto* exp_avg_buffer = exp_avg.buffer();
    auto* exp_avg_sq_buffer = exp_avg_sq.buffer();
    auto* max_exp_avg_sq_buffer = max_exp_avg_sq_opt.has_value() ? max_exp_avg_sq_opt.value().buffer() : nullptr;
    auto* output_buffer = output.buffer();

    std::map<std::string, std::string> defines;
    defines["AMSGRAD"] = amsgrad ? "1" : "0";
    defines["STOCH_ROUND"] = "0";  // No stochastic rounding for full precision

    AdamWFullPrecisionKernels kernels{};

    std::vector<uint32_t> reader_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(param_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(grad_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_sq_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(max_exp_avg_sq_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_sq_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(max_exp_avg_sq_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for full precision adamw
    // -------------------------------------------------------------------------

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::UnpackToDestFp32);
    unpack_to_dest_mode[kMomentumCbIndex] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kVarianceCbIndex] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kMaxExpAvgSqCbIndex] = UnpackToDestMode::UnpackToDestFp32;

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        block_size};                 // per_core_block_size

    tt::tt_metal::ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = false,
        .compile_args = compute_group_1_args,
        .defines = defines,
    };
    kernels.compute_group_1 = tt::tt_metal::CreateKernel(program, kComputeKernelPath, core_group_1, compute_config);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_tiles_per_core_group_2, block_size};
        compute_config.compile_args = compute_group_2_args;
        kernels.compute_group_2 = tt::tt_metal::CreateKernel(program, kComputeKernelPath, core_group_2, compute_config);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------

    assign_per_core_runtime_args_full_precision(
        program,
        kernels,
        param_buffer,
        grad_buffer,
        exp_avg_buffer,
        exp_avg_sq_buffer,
        max_exp_avg_sq_buffer,
        lr,
        beta1,
        beta2,
        beta1_pow,
        beta2_pow,
        epsilon,
        weight_decay,
        step,
        output_buffer,
        num_cores,
        num_cores_y,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        core_group_1,
        core_group_2);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id  = */ kernels.reader,
         /* writer_kernel_id  = */ kernels.writer,
         /* compute_kernel_group_1_id = */ kernels.compute_group_1,
         /* compute_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void AdamWFullPrecisionProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& reader_kernel_id = shared_variables.reader_kernel_id;
    auto& writer_kernel_id = shared_variables.writer_kernel_id;
    auto& compute_kernel_group_1_id = shared_variables.compute_kernel_group_1_id;
    auto& compute_kernel_group_2_id = shared_variables.compute_kernel_group_2_id;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* param_buffer = tensor_args.param.buffer();
    auto* grad_buffer = tensor_args.grad.buffer();
    auto* exp_avg_buffer = tensor_args.exp_avg.buffer();
    auto* exp_avg_sq_buffer = tensor_args.exp_avg_sq.buffer();
    auto* max_exp_avg_sq_buffer =
        tensor_args.max_exp_avg_sq.has_value() ? tensor_args.max_exp_avg_sq.value().buffer() : nullptr;

    auto lr = operation_attributes.lr;
    auto beta1 = operation_attributes.beta1;
    auto beta2 = operation_attributes.beta2;
    auto beta1_pow = operation_attributes.beta1_pow;
    auto beta2_pow = operation_attributes.beta2_pow;
    auto epsilon = operation_attributes.epsilon;
    auto weight_decay = operation_attributes.weight_decay;
    auto* output_buffer = tensor_return_value.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    float bias_correction1 = 1.0f - beta1_pow;
    float bias_correction2 = 1.0f - beta2_pow;
    float step_size = lr / bias_correction1;
    float inv_sqrt_bc2 = 1.0f / std::sqrt(bias_correction2);
    float decay_factor = 1.0f - lr * weight_decay;

    // Update:
    // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))

    auto& compute_group_1_common_args = GetCommonRuntimeArgs(program, compute_kernel_group_1_id);
    compute_group_1_common_args[kComputeBeta1Idx] = std::bit_cast<uint32_t>(beta1);
    compute_group_1_common_args[kComputeBeta2Idx] = std::bit_cast<uint32_t>(beta2);
    compute_group_1_common_args[kComputeEpsilonIdx] = std::bit_cast<uint32_t>(epsilon);
    compute_group_1_common_args[kComputeStepSizeIdx] = std::bit_cast<uint32_t>(step_size);
    compute_group_1_common_args[kComputeInvSqrtBiasCorrection2Idx] = std::bit_cast<uint32_t>(inv_sqrt_bc2);
    compute_group_1_common_args[kComputeOneMinusBeta1Idx] = std::bit_cast<uint32_t>(one_minus_beta1);
    compute_group_1_common_args[kComputeOneMinusBeta2Idx] = std::bit_cast<uint32_t>(one_minus_beta2);
    compute_group_1_common_args[kComputeDecayFactorIdx] = std::bit_cast<uint32_t>(decay_factor);
    compute_group_1_common_args[kComputeSeedIdx] = 0U;  // No stochastic rounding

    if (!core_group_2.ranges().empty()) {
        auto& compute_group_2_common_args = GetCommonRuntimeArgs(program, compute_kernel_group_2_id);
        compute_group_2_common_args[kComputeBeta1Idx] = std::bit_cast<uint32_t>(beta1);
        compute_group_2_common_args[kComputeBeta2Idx] = std::bit_cast<uint32_t>(beta2);
        compute_group_2_common_args[kComputeEpsilonIdx] = std::bit_cast<uint32_t>(epsilon);
        compute_group_2_common_args[kComputeStepSizeIdx] = std::bit_cast<uint32_t>(step_size);
        compute_group_2_common_args[kComputeInvSqrtBiasCorrection2Idx] = std::bit_cast<uint32_t>(inv_sqrt_bc2);
        compute_group_2_common_args[kComputeOneMinusBeta1Idx] = std::bit_cast<uint32_t>(one_minus_beta1);
        compute_group_2_common_args[kComputeOneMinusBeta2Idx] = std::bit_cast<uint32_t>(one_minus_beta2);
        compute_group_2_common_args[kComputeDecayFactorIdx] = std::bit_cast<uint32_t>(decay_factor);
        compute_group_2_common_args[kComputeSeedIdx] = 0U;  // No stochastic rounding
    }

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kParamAddrIdx] = param_buffer->address();
            runtime_args[kGradAddrIdx] = grad_buffer->address();
            runtime_args[kExpAvgAddrIdx] = exp_avg_buffer->address();
            runtime_args[kExpAvgSqAddrIdx] = exp_avg_sq_buffer->address();
            runtime_args[kMaxExpAvgSqAddrIdx] =
                max_exp_avg_sq_buffer != nullptr ? max_exp_avg_sq_buffer->address() : 0U;
        }
        // Update output buffer for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputAddrIdx] = output_buffer->address();
            runtime_args[kExpAvgAddrIdxOut] = exp_avg_buffer->address();
            runtime_args[kExpAvgSqAddrIdxOut] = exp_avg_sq_buffer->address();
            runtime_args[kMaxExpAvgSqAddrIdxOut] =
                max_exp_avg_sq_buffer != nullptr ? max_exp_avg_sq_buffer->address() : 0U;
        }
    }
}

}  // namespace ttml::metal::optimizers::adamw::device
