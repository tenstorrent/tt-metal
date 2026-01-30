// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_program_factory.hpp"

#include <common/TracyQueue.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <random>
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
struct AdamWKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const AdamWKernels& kernels,
    const tt::tt_metal::Buffer* param_buffer,
    const tt::tt_metal::Buffer* grad_buffer,
    const tt::tt_metal::Buffer* exp_avg_buffer,
    const tt::tt_metal::Buffer* exp_avg_sq_buffer,
    const tt::tt_metal::Buffer* max_exp_avg_sq_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const operation_attributes_t& attrs,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    float one_minus_beta1 = 1.0f - attrs.beta1;
    float one_minus_beta2 = 1.0f - attrs.beta2;

    float bias_correction1 = 1.0f - attrs.beta1_pow;
    float bias_correction2 = 1.0f - attrs.beta2_pow;
    float step_size = attrs.lr / bias_correction1;
    float inv_sqrt_bc2 = 1.0f / std::sqrt(bias_correction2);
    float decay_factor = 1.0f - attrs.lr * attrs.weight_decay;

    // Generate seeds for stochastic rounding (0 if disabled)
    std::vector<uint32_t> seeds(num_cores, 0);
    if (attrs.stochastic_rounding == StochasticRounding::Enabled) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(1, 0xFFFFFFFF);
        for (uint32_t i = 0; i < num_cores; i++) {
            seeds[i] = dis(gen);
        }
    }

    // Compute runtime args (same for all cores except seed)
    std::vector<uint32_t> compute_args{
        std::bit_cast<uint32_t>(attrs.beta1),
        std::bit_cast<uint32_t>(attrs.beta2),
        std::bit_cast<uint32_t>(attrs.epsilon),
        std::bit_cast<uint32_t>(step_size),
        std::bit_cast<uint32_t>(inv_sqrt_bc2),
        std::bit_cast<uint32_t>(one_minus_beta1),
        std::bit_cast<uint32_t>(one_minus_beta2),
        std::bit_cast<uint32_t>(decay_factor),
        0U  // seed placeholder, updated per-core
    };

    // Update:
    // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))

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

        // Compute kernel
        compute_args[kComputeSeedIdx] = seeds[i];
        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, kernels.compute_group_1, core, compute_args);
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(program, kernels.compute_group_2, core, compute_args);
        } else {
            TT_THROW("Core {} not in specified core ranges", core);
        }

        // Writer kernel: (param_out_addr, exp_avg_addr, exp_avg_sq_addr, max_exp_avg_sq_addr, number_of_tiles,
        // offset_in_tiles)
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

AdamWProgramFactory::cached_program_t AdamWProgramFactory::create(
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

    auto* device = param.device();

    tt::tt_metal::Program program{};

    // Determine data formats based on param dtype
    tt::DataFormat param_data_format = datatype_to_dataformat_converter(param.dtype());
    tt::DataFormat grad_data_format = tt::DataFormat::Float16_b;  // Gradient is always bf16
    tt::DataFormat intermediate_data_format = tt::DataFormat::Float32;

    uint32_t param_single_tile_size_bytes = tt::tile_size(param_data_format);
    uint32_t grad_single_tile_size_bytes = tt::tile_size(grad_data_format);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

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

    // param CB - uses param_data_format (bf16 or fp32 depending on mode)
    [[maybe_unused]] auto cb_param = create_circular_buffer(
        program, all_cores, kParamCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles);

    // grad CB - always bf16
    [[maybe_unused]] auto cb_grad = create_circular_buffer(
        program, all_cores, kGradCbIndex, grad_data_format, grad_single_tile_size_bytes, num_input_tiles);

    // exp_avg and exp_avg_sq CBs - use param_data_format (bf16 or fp32)
    [[maybe_unused]] auto cb_exp_avg = create_circular_buffer(
        program, all_cores, kExpAvgCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_exp_avg_sq = create_circular_buffer(
        program, all_cores, kExpAvgSqCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles);

    // output CB - uses param_data_format
    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_exp_avg_out = create_circular_buffer(
        program, all_cores, kExpAvgOutCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles);

    [[maybe_unused]] auto cb_exp_avg_sq_out = create_circular_buffer(
        program, all_cores, kExpAvgSqOutCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles);

    // AMSGrad-specific CBs - only create if amsgrad is enabled
    if (operation_attributes.amsgrad) {
        [[maybe_unused]] auto cb_max_exp_avg_sq_in = create_circular_buffer(
            program,
            all_cores,
            kMaxExpAvgSqInCbIndex,
            param_data_format,
            param_single_tile_size_bytes,
            num_input_tiles);

        [[maybe_unused]] auto cb_max_exp_avg_sq_out = create_circular_buffer(
            program,
            all_cores,
            kMaxExpAvgSqOutCbIndex,
            param_data_format,
            param_single_tile_size_bytes,
            num_output_tiles);

        [[maybe_unused]] auto cb_max_exp_avg_sq = create_circular_buffer(
            program,
            all_cores,
            kMaxExpAvgSqCbIndex,
            intermediate_data_format,
            float32_single_tile_size_bytes,
            num_output_tiles);
    }

    // Intermediate CBs are always fp32
    [[maybe_unused]] auto cb_momentum = create_circular_buffer(
        program,
        all_cores,
        kMomentumCbIndex,
        intermediate_data_format,
        float32_single_tile_size_bytes,
        num_output_tiles);

    [[maybe_unused]] auto cb_variance = create_circular_buffer(
        program,
        all_cores,
        kVarianceCbIndex,
        intermediate_data_format,
        float32_single_tile_size_bytes,
        num_output_tiles);

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
    defines["AMSGRAD"] = operation_attributes.amsgrad ? "1" : "0";
    defines["STOCH_ROUND"] = operation_attributes.stochastic_rounding == StochasticRounding::Enabled ? "1" : "0";

    AdamWKernels kernels{};

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
    // 4) Create compute kernels for fused adamw
    // -------------------------------------------------------------------------

    // FPU is not used at all in the operation
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::UnpackToDestFp32);

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

    assign_per_core_runtime_args(
        program,
        kernels,
        param_buffer,
        grad_buffer,
        exp_avg_buffer,
        exp_avg_sq_buffer,
        max_exp_avg_sq_buffer,
        output_buffer,
        operation_attributes,
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

void AdamWProgramFactory::override_runtime_arguments(
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
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    const auto& attrs = operation_attributes;

    auto* param_buffer = tensor_args.param.buffer();
    auto* grad_buffer = tensor_args.grad.buffer();
    auto* exp_avg_buffer = tensor_args.exp_avg.buffer();
    auto* exp_avg_sq_buffer = tensor_args.exp_avg_sq.buffer();
    auto* max_exp_avg_sq_buffer =
        tensor_args.max_exp_avg_sq.has_value() ? tensor_args.max_exp_avg_sq.value().buffer() : nullptr;
    auto* output_buffer = tensor_return_value.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);
    auto& compute_group_1_runtime_args = GetRuntimeArgs(program, compute_kernel_group_1_id);
    auto& compute_group_2_runtime_args = core_group_2.ranges().empty()
                                             ? compute_group_1_runtime_args
                                             : GetRuntimeArgs(program, compute_kernel_group_2_id);

    float one_minus_beta1 = 1.0f - attrs.beta1;
    float one_minus_beta2 = 1.0f - attrs.beta2;

    float bias_correction1 = 1.0f - attrs.beta1_pow;
    float bias_correction2 = 1.0f - attrs.beta2_pow;
    float step_size = attrs.lr / bias_correction1;
    float inv_sqrt_bc2 = 1.0f / std::sqrt(bias_correction2);
    float decay_factor = 1.0f - attrs.lr * attrs.weight_decay;

    // Generate seeds for stochastic rounding (0 if disabled)
    std::vector<uint32_t> seeds(num_cores, 0);
    if (attrs.stochastic_rounding == StochasticRounding::Enabled) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(1, 0xFFFFFFFF);
        for (uint32_t i = 0; i < num_cores; i++) {
            seeds[i] = dis(gen);
        }
    }

    // Helper to select correct compute runtime args with error handling
    auto get_compute_runtime_args = [&](const tt::tt_metal::CoreCoord& core) -> auto& {
        if (core_group_1.contains(core)) {
            return compute_group_1_runtime_args[core.x][core.y];
        }
        if (core_group_2.contains(core)) {
            return compute_group_2_runtime_args[core.x][core.y];
        }
        TT_THROW("Core {} not in specified core ranges", core);
    };

    // Update:
    // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))
    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update reader kernel args
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kParamAddrIdx] = param_buffer->address();
            runtime_args[kGradAddrIdx] = grad_buffer->address();
            runtime_args[kExpAvgAddrIdx] = exp_avg_buffer->address();
            runtime_args[kExpAvgSqAddrIdx] = exp_avg_sq_buffer->address();
            runtime_args[kMaxExpAvgSqAddrIdx] =
                max_exp_avg_sq_buffer != nullptr ? max_exp_avg_sq_buffer->address() : 0U;
        }
        // Update compute kernel args
        {
            auto& runtime_args = get_compute_runtime_args(core);
            runtime_args[kComputeBeta1Idx] = std::bit_cast<uint32_t>(attrs.beta1);
            runtime_args[kComputeBeta2Idx] = std::bit_cast<uint32_t>(attrs.beta2);
            runtime_args[kComputeEpsilonIdx] = std::bit_cast<uint32_t>(attrs.epsilon);
            runtime_args[kComputeStepSizeIdx] = std::bit_cast<uint32_t>(step_size);
            runtime_args[kComputeInvSqrtBiasCorrection2Idx] = std::bit_cast<uint32_t>(inv_sqrt_bc2);
            runtime_args[kComputeOneMinusBeta1Idx] = std::bit_cast<uint32_t>(one_minus_beta1);
            runtime_args[kComputeOneMinusBeta2Idx] = std::bit_cast<uint32_t>(one_minus_beta2);
            runtime_args[kComputeDecayFactorIdx] = std::bit_cast<uint32_t>(decay_factor);
            runtime_args[kComputeSeedIdx] = seeds[i];
        }
        // Update writer kernel args
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
