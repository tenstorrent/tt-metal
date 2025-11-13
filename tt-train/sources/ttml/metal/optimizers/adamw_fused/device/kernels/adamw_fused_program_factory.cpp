// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_fused_program_factory.hpp"

#include <common/TracyQueue.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "adamw_fused_device_operation_types.hpp"
#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw_fused/device/kernels/dataflow/"
    "reader_adamw_fused_interleaved_start_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw_fused/device/kernels/dataflow/"
    "writer_adamw_fused_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/optimizers/adamw_fused/device/kernels/compute/adamw_fused_kernel.cpp";

// reader runtime args
constexpr uint32_t kParamAddrIdx = 0;
constexpr uint32_t kGradAddrIdx = 1U;
constexpr uint32_t kFirstMomentAddrIdx = 2U;
constexpr uint32_t kSecondMomentAddrIdx = 3U;
constexpr uint32_t kLrIdx = 4U;
constexpr uint32_t kBeta1Idx = 5U;
constexpr uint32_t kBeta2Idx = 6U;
constexpr uint32_t kEpsilonIdx = 7U;
constexpr uint32_t kWeightDecayIdx = 8U;
// compute runtime args
constexpr uint32_t kComputeLrIdx = 0U;
constexpr uint32_t kComputeBeta1Idx = 1U;
constexpr uint32_t kComputeBeta2Idx = 2U;
constexpr uint32_t kComputeEpsilonIdx = 3U;
constexpr uint32_t kComputeWeightDecayIdx = 4U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0;
constexpr uint32_t kFirstMomentAddrIdxOut = 1U;
constexpr uint32_t kSecondMomentAddrIdxOut = 2U;

constexpr auto kParamCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kFirstMomentCbIndex = tt::CBIndex::c_2;
constexpr auto kSecondMomentCbIndex = tt::CBIndex::c_3;

constexpr auto kOutputCbIndex = tt::CBIndex::c_16;

}  // namespace

namespace ttml::metal::optimizers::adamw_fused::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct AdamWFusedKernels {
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
    const AdamWFusedKernels& kernels,
    const tt::tt_metal::Buffer* param_buffer,
    const tt::tt_metal::Buffer* grad_buffer,
    const tt::tt_metal::Buffer* first_moment_buffer,
    const tt::tt_metal::Buffer* second_moment_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const uint32_t step,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many tiles this core will process
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core {} not in specified core ranges", core);
        }

        // Reader kernel: (param_addr, grad_addr, first_moment_addr, second_moment_addr, lr, beta1, beta2, epsilon,
        // weight_decay, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {param_buffer->address(),
             grad_buffer->address(),
             first_moment_buffer->address(),
             second_moment_buffer->address(),
             std::bit_cast<uint32_t>(lr),
             std::bit_cast<uint32_t>(beta1),
             std::bit_cast<uint32_t>(beta2),
             std::bit_cast<uint32_t>(epsilon),
             std::bit_cast<uint32_t>(weight_decay),
             num_tiles_per_core,
             num_tiles_written});

        // Compute kernel:
        if (core_group_1.contains(core)) {
            SetRuntimeArgs(
                program,
                kernels.compute_group_1,
                core,
                {std::bit_cast<uint32_t>(lr),
                 std::bit_cast<uint32_t>(beta1),
                 std::bit_cast<uint32_t>(beta2),
                 std::bit_cast<uint32_t>(epsilon),
                 std::bit_cast<uint32_t>(weight_decay)});
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(
                program,
                kernels.compute_group_2,
                core,
                {std::bit_cast<uint32_t>(lr),
                 std::bit_cast<uint32_t>(beta1),
                 std::bit_cast<uint32_t>(beta2),
                 std::bit_cast<uint32_t>(epsilon),
                 std::bit_cast<uint32_t>(weight_decay)});
        } else {
            TT_THROW("Core {} not in specified core ranges", core);
        }

        // Writer kernel: (param_out_addr, first_moment_addr, second_moment_addr, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(),
             first_moment_buffer->address(),
             second_moment_buffer->address(),
             num_tiles_per_core,
             num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }
}

AdamWFusedProgramFactory::cached_program_t AdamWFusedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& param = tensor_args.param;
    const auto& grad = tensor_args.grad;
    const auto& first_moment = tensor_args.first_moment;
    const auto& second_moment = tensor_args.second_moment;
    const auto& lr = operation_attributes.lr;
    const auto& beta1 = operation_attributes.beta1;
    const auto& beta2 = operation_attributes.beta2;
    const auto& epsilon = operation_attributes.epsilon;
    const auto& weight_decay = operation_attributes.weight_decay;
    const auto& step = operation_attributes.step;

    auto* device = param.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(param.dtype());
    tt::DataFormat output_data_format = input_data_format;
    [[maybe_unused]] tt::DataFormat intermediate_data_format = tt::DataFormat::Float32;

    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");
    TT_FATAL(output_data_format == tt::DataFormat::Float16_b, "Output data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    [[maybe_unused]] uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

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

    uint32_t block_size = std::min(4U, num_tiles_per_core_group_1);
    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t twice_block_size = 2U * block_size;

    const uint32_t num_input_tiles = twice_block_size;
    const uint32_t num_output_tiles = twice_block_size;

    [[maybe_unused]] auto cb_param = create_circular_buffer(
        program, all_cores, kParamCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_grad = create_circular_buffer(
        program, all_cores, kGradCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_first_moment = create_circular_buffer(
        program, all_cores, kFirstMomentCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_second_moment = create_circular_buffer(
        program, all_cores, kSecondMomentCbIndex, input_data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, output_data_format, bfloat16_single_tile_size_bytes, num_output_tiles);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* param_buffer = param.buffer();
    auto* grad_buffer = grad.buffer();
    auto* first_moment_buffer = first_moment.buffer();
    auto* second_moment_buffer = second_moment.buffer();
    auto* output_buffer = output.buffer();

    AdamWFusedKernels kernels{};

    std::vector<uint32_t> reader_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(param_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(grad_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(first_moment_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(second_moment_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(first_moment_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(second_moment_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for fused adamw
    // -------------------------------------------------------------------------
    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        block_size};                 // per_core_block_size

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_tiles_per_core_group_2, block_size};
        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------

    assign_per_core_runtime_args(
        program,
        kernels,
        param_buffer,
        grad_buffer,
        first_moment_buffer,
        second_moment_buffer,
        lr,
        beta1,
        beta2,
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
        {/* adamw_fused_reader_kernel_id  = */ kernels.reader,
         /* adamw_fused_writer_kernel_id  = */ kernels.writer,
         /* adamw_fused_kernel_group_1_id = */ kernels.compute_group_1,
         /* adamw_fused_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void AdamWFusedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& adamw_fused_reader_kernel_id = shared_variables.reader_kernel_id;
    auto& adamw_fused_writer_kernel_id = shared_variables.writer_kernel_id;
    auto& adamw_fused_compute_kernel_group_1_id = shared_variables.compute_kernel_group_1_id;
    auto& adamw_fused_compute_kernel_group_2_id = shared_variables.compute_kernel_group_2_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* param_buffer = tensor_args.param.buffer();
    auto* grad_buffer = tensor_args.grad.buffer();
    auto* first_moment_buffer = tensor_args.first_moment.buffer();
    auto* second_moment_buffer = tensor_args.second_moment.buffer();

    auto lr = operation_attributes.lr;
    auto beta1 = operation_attributes.beta1;
    auto beta2 = operation_attributes.beta2;
    auto epsilon = operation_attributes.epsilon;
    auto weight_decay = operation_attributes.weight_decay;
    auto step = operation_attributes.step;
    auto* output_buffer = tensor_return_value.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, adamw_fused_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, adamw_fused_writer_kernel_id);
    auto& compute_group_1_runtime_args = GetRuntimeArgs(program, adamw_fused_compute_kernel_group_1_id);
    [[maybe_unused]] auto& compute_group_2_runtime_args =
        core_group_2.ranges().empty() ? compute_group_1_runtime_args
                                      : GetRuntimeArgs(program, adamw_fused_compute_kernel_group_2_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kParamAddrIdx] = param_buffer->address();
            runtime_args[kGradAddrIdx] = grad_buffer->address();
            runtime_args[kFirstMomentAddrIdx] = first_moment_buffer->address();
            runtime_args[kSecondMomentAddrIdx] = second_moment_buffer->address();
            runtime_args[kLrIdx] = std::bit_cast<uint32_t>(lr);
            runtime_args[kBeta1Idx] = std::bit_cast<uint32_t>(beta1);
            runtime_args[kBeta2Idx] = std::bit_cast<uint32_t>(beta2);
            runtime_args[kEpsilonIdx] = std::bit_cast<uint32_t>(epsilon);
            runtime_args[kWeightDecayIdx] = std::bit_cast<uint32_t>(weight_decay);
        }
        if (core_group_1.contains(core)) {
            [[maybe_unused]] auto& runtime_args = compute_group_1_runtime_args[core.x][core.y];
            runtime_args[kComputeLrIdx] = std::bit_cast<uint32_t>(lr);
            runtime_args[kComputeBeta1Idx] = std::bit_cast<uint32_t>(beta1);
            runtime_args[kComputeBeta2Idx] = std::bit_cast<uint32_t>(beta2);
            runtime_args[kComputeEpsilonIdx] = std::bit_cast<uint32_t>(epsilon);
            runtime_args[kComputeWeightDecayIdx] = std::bit_cast<uint32_t>(weight_decay);
        } else if (core_group_2.contains(core)) {
            [[maybe_unused]] auto& runtime_args = compute_group_2_runtime_args[core.x][core.y];
            runtime_args[kComputeLrIdx] = std::bit_cast<uint32_t>(lr);
            runtime_args[kComputeBeta1Idx] = std::bit_cast<uint32_t>(beta1);
            runtime_args[kComputeBeta2Idx] = std::bit_cast<uint32_t>(beta2);
            runtime_args[kComputeEpsilonIdx] = std::bit_cast<uint32_t>(epsilon);
            runtime_args[kComputeWeightDecayIdx] = std::bit_cast<uint32_t>(weight_decay);
        } else {
            TT_THROW("Core {} not in specified core ranges", core);
        }
        // Update output buffer for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputAddrIdx] = output_buffer->address();
            runtime_args[kFirstMomentAddrIdxOut] = first_moment_buffer->address();
            runtime_args[kSecondMomentAddrIdxOut] = second_moment_buffer->address();
        }
    }
}

}  // namespace ttml::metal::optimizers::adamw_fused::device
