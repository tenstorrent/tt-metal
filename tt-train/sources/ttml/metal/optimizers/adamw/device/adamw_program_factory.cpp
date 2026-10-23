// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_program_factory.hpp"

#include <common/TracyQueue.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "adamw_device_operation.hpp"
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

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute group 1, then compute
// group 2 when it has cores.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;
constexpr uint32_t kComputeGroup1KernelIdx = 2U;
constexpr uint32_t kComputeGroup2KernelIdx = 3U;

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

namespace {

constexpr uint32_t kGoldenRatio32 = 0x9E3779B9U;

// splitmix32 finalizer -- decorrelates seeds derived from one base draw.
constexpr uint32_t mix_seed(uint32_t x) {
    x = (x ^ (x >> 16)) * 0x21F0AAADU;
    x = (x ^ (x >> 15)) * 0x735A2D97U;
    return x ^ (x >> 15);
}

// 0xFFFFFFFF is the XNOR LFSR lock-up state; 0 makes the first ~10 draws per lane zero.
constexpr bool is_valid_prng_seed(uint32_t seed) {
    return seed != 0U && seed != 0xFFFFFFFFU;
}

// Salt for re-hashing an invalid seed.
constexpr uint32_t kSeedRemapSalt = 0x5BD1E995U;

constexpr uint32_t make_core_seed(uint32_t base, uint32_t core_idx) {
    uint32_t seed = mix_seed(base + core_idx * kGoldenRatio32);
    while (!is_valid_prng_seed(seed)) {
        seed = mix_seed(seed ^ kSeedRemapSalt);
    }
    return seed;
}

// Compute runtime arg 8 of the core at `core_index` in the walk; 0 when no seed is passed.
constexpr uint32_t make_stochastic_rounding_seed(std::optional<uint32_t> base, uint32_t core_index) {
    return base.has_value() ? make_core_seed(*base, core_index) : 0U;
}

// Compute runtime args 0-7, all derived from hash-excluded attributes. create_descriptor() and
// override_runtime_arguments() both take them from here, so a cache hit writes exactly what a rebuild would.
struct ComputeScalarArgs {
    uint32_t beta1;
    uint32_t beta2;
    uint32_t epsilon;
    uint32_t step_size;
    uint32_t inv_sqrt_bc2;
    uint32_t one_minus_beta1;
    uint32_t one_minus_beta2;
    uint32_t decay_factor;
};

// Update:
// theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))
ComputeScalarArgs make_compute_scalar_args(const operation_attributes_t& attrs) {
    const float one_minus_beta1 = 1.0f - attrs.beta1;
    const float one_minus_beta2 = 1.0f - attrs.beta2;
    const float bias_correction1 = 1.0f - attrs.beta1_pow;
    const float bias_correction2 = 1.0f - attrs.beta2_pow;
    const float step_size = attrs.lr / bias_correction1;
    const float inv_sqrt_bc2 = 1.0f / std::sqrt(bias_correction2);
    const float decay_factor = 1.0f - attrs.lr * attrs.weight_decay;
    return ComputeScalarArgs{
        .beta1 = std::bit_cast<uint32_t>(attrs.beta1),
        .beta2 = std::bit_cast<uint32_t>(attrs.beta2),
        .epsilon = std::bit_cast<uint32_t>(attrs.epsilon),
        .step_size = std::bit_cast<uint32_t>(step_size),
        .inv_sqrt_bc2 = std::bit_cast<uint32_t>(inv_sqrt_bc2),
        .one_minus_beta1 = std::bit_cast<uint32_t>(one_minus_beta1),
        .one_minus_beta2 = std::bit_cast<uint32_t>(one_minus_beta2),
        .decay_factor = std::bit_cast<uint32_t>(decay_factor),
    };
}

// The param's padded tiles over the device's compute grid. create_descriptor() pins it against split_work_to_cores,
// and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& param) {
    return compute_work_split(
        param.device()->compute_with_storage_grid_size(),
        static_cast<uint32_t>(param.physical_volume() / tt::constants::TILE_HW));
}

}  // namespace

tt::tt_metal::ProgramDescriptor AdamWProgramFactory::create_descriptor(
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

    tt::tt_metal::ProgramDescriptor desc;

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

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles_to_process);
    const WorkSplit split = make_work_split(param);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);

    uint32_t block_size = std::min(2U, num_tiles_per_core_group_1);
    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t intermediate_num_tiles = twice_block_size;

    // Use 1× block_size pipeline depth: benchmarks at tt-train/tests/benchmark/adamw_benchmark.cpp proved it to be the
    // best for BW utilization
    uint32_t pipeline_depth = block_size;

    const uint32_t num_input_tiles = pipeline_depth;
    const uint32_t num_output_tiles = pipeline_depth;

    // param CB - uses param_data_format (bf16 or fp32 depending on mode)
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kParamCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles));

    // grad CB - always bf16
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGradCbIndex, grad_data_format, grad_single_tile_size_bytes, num_input_tiles));

    // exp_avg and exp_avg_sq CBs - use param_data_format (bf16 or fp32)
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kExpAvgCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kExpAvgSqCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles));

    // output CB - uses param_data_format
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kOutputCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kExpAvgOutCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kExpAvgSqOutCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles));

    // AMSGrad-specific CBs - only create if amsgrad is enabled
    if (operation_attributes.amsgrad) {
        desc.cbs.push_back(make_cb_descriptor(
            all_cores, kMaxExpAvgSqInCbIndex, param_data_format, param_single_tile_size_bytes, num_input_tiles));

        desc.cbs.push_back(make_cb_descriptor(
            all_cores, kMaxExpAvgSqOutCbIndex, param_data_format, param_single_tile_size_bytes, num_output_tiles));

        desc.cbs.push_back(make_cb_descriptor(
            all_cores,
            kMaxExpAvgSqCbIndex,
            intermediate_data_format,
            float32_single_tile_size_bytes,
            intermediate_num_tiles));
    }

    // Intermediate CBs are always fp32
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMomentumCbIndex, intermediate_data_format, float32_single_tile_size_bytes, intermediate_num_tiles));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kVarianceCbIndex, intermediate_data_format, float32_single_tile_size_bytes, intermediate_num_tiles));

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

    std::vector<uint32_t> reader_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(param_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(grad_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_sq_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(max_exp_avg_sq_buffer).append_to(reader_compile_time_args);
    desc.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(exp_avg_sq_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(max_exp_avg_sq_buffer).append_to(writer_compile_time_args);
    desc.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for fused adamw
    // -------------------------------------------------------------------------

    // FPU is not used at all in the operation
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        block_size};                 // per_core_block_size

    desc.kernels.push_back(make_compute_kernel_descriptor(  // kComputeGroup1KernelIdx
        core_group_1,
        compute_group_1_args,
        defines,
        kComputeKernelPath,
        /*fp32_dest_acc_en=*/true,
        unpack_to_dest_mode));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_tiles_per_core_group_2, block_size};
        desc.kernels.push_back(make_compute_kernel_descriptor(  // kComputeGroup2KernelIdx
            core_group_2,
            compute_group_2_args,
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true,
            unpack_to_dest_mode));
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses, the scalars and the
    // seeds on every cache hit. num_tiles/start_tile follow from the hashed param shape.
    const ComputeScalarArgs scalars = make_compute_scalar_args(operation_attributes);
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_tiles, start_tile, in_group_1] = work;
        reader.emplace_runtime_args(
            core,
            {param_buffer,
             grad_buffer,
             exp_avg_buffer,
             exp_avg_sq_buffer,
             max_exp_avg_sq_buffer,  // null (0) when max_exp_avg_sq is absent
             num_tiles,
             start_tile});
        desc.kernels[in_group_1 ? kComputeGroup1KernelIdx : kComputeGroup2KernelIdx].emplace_runtime_args(
            core,
            {scalars.beta1,
             scalars.beta2,
             scalars.epsilon,
             scalars.step_size,
             scalars.inv_sqrt_bc2,
             scalars.one_minus_beta1,
             scalars.one_minus_beta2,
             scalars.decay_factor,
             make_stochastic_rounding_seed(operation_attributes.stochastic_rounding_seed, core_index)});
        writer.emplace_runtime_args(
            core, {output_buffer, exp_avg_buffer, exp_avg_sq_buffer, max_exp_avg_sq_buffer, num_tiles, start_tile});
    });

    // -------------------------------------------------------------------------
    // 6) Return the program descriptor
    // -------------------------------------------------------------------------
    return desc;
}

void AdamWProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Nothing else runs on a cache hit, so every address, scalar and seed is rewritten on every core. The tile counts
    // and offsets (reader 5-6, writer 4-5) follow from the hashed param shape and keep their create_descriptor() value.
    const WorkSplit split = make_work_split(tensor_args.param);

    const uint32_t param_addr = tensor_args.param.buffer()->address();
    const uint32_t grad_addr = tensor_args.grad.buffer()->address();
    const uint32_t exp_avg_addr = tensor_args.exp_avg.buffer()->address();
    const uint32_t exp_avg_sq_addr = tensor_args.exp_avg_sq.buffer()->address();
    const uint32_t max_exp_avg_sq_addr =
        tensor_args.max_exp_avg_sq.has_value() ? tensor_args.max_exp_avg_sq->buffer()->address() : 0U;
    const uint32_t output_addr = tensor_return_value.buffer()->address();

    const ComputeScalarArgs scalars = make_compute_scalar_args(operation_attributes);
    const std::optional<uint32_t> seed_base = operation_attributes.stochastic_rounding_seed;

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    auto& compute_group_1_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kComputeGroup1KernelIdx);
    // Without group 2 there is no fourth kernel and every core is in group 1, so the alias is never indexed.
    auto& compute_group_2_runtime_args = split.has_group_2()
                                             ? tt::tt_metal::GetRuntimeArgs(program, kComputeGroup2KernelIdx)
                                             : compute_group_1_runtime_args;

    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kParamAddrIdx] = param_addr;
        reader_args[kGradAddrIdx] = grad_addr;
        reader_args[kExpAvgAddrIdx] = exp_avg_addr;
        reader_args[kExpAvgSqAddrIdx] = exp_avg_sq_addr;
        reader_args[kMaxExpAvgSqAddrIdx] = max_exp_avg_sq_addr;

        auto& compute_args =
            (work.in_group_1 ? compute_group_1_runtime_args : compute_group_2_runtime_args)[core.x][core.y];
        compute_args[kComputeBeta1Idx] = scalars.beta1;
        compute_args[kComputeBeta2Idx] = scalars.beta2;
        compute_args[kComputeEpsilonIdx] = scalars.epsilon;
        compute_args[kComputeStepSizeIdx] = scalars.step_size;
        compute_args[kComputeInvSqrtBiasCorrection2Idx] = scalars.inv_sqrt_bc2;
        compute_args[kComputeOneMinusBeta1Idx] = scalars.one_minus_beta1;
        compute_args[kComputeOneMinusBeta2Idx] = scalars.one_minus_beta2;
        compute_args[kComputeDecayFactorIdx] = scalars.decay_factor;
        compute_args[kComputeSeedIdx] = make_stochastic_rounding_seed(seed_base, work.index);

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kOutputAddrIdx] = output_addr;
        writer_args[kExpAvgAddrIdxOut] = exp_avg_addr;
        writer_args[kExpAvgSqAddrIdxOut] = exp_avg_sq_addr;
        writer_args[kMaxExpAvgSqAddrIdxOut] = max_exp_avg_sq_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(AdamWDeviceOperation, AdamWProgramFactory);

}  // namespace ttml::metal::optimizers::adamw::device
