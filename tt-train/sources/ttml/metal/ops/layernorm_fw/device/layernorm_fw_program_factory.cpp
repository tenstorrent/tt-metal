// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/dataflow/writer_layernorm_fw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/dataflow/reader_layernorm_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_fw/device/kernels/compute/layernorm_fw_kernel.cpp";

// Reader buffer indices
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kGammaBufferIdx = 1U;
constexpr uint32_t kBetaBufferIdx = 2U;

// Writer buffer indices
constexpr uint32_t kOutputBufferIdx = 0;
constexpr uint32_t kMeanBufferIdx = 1U;
constexpr uint32_t kRstdBufferIdx = 2U;

// CBs with input data
constexpr auto kScalerCbIndex = tt::CBIndex::c_0;  // 1/N scaler
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;   // mask for width dimension
constexpr auto kEpsCbIndex = tt::CBIndex::c_2;     // epsilon
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;   // gamma (scale parameter)
constexpr auto kBetaCbIndex = tt::CBIndex::c_4;    // beta (shift parameter)
constexpr auto kInputCbIndex = tt::CBIndex::c_5;   // input tensor

// CBs with output data
constexpr auto kOutputCbIndex = tt::CBIndex::c_6;  // normalized output
constexpr auto kMeanCbIndex = tt::CBIndex::c_7;    // mean (for backward pass)
constexpr auto kRstdCbIndex = tt::CBIndex::c_8;    // rstd (for backward pass)

// CBs with intermediate computations
constexpr auto kSumCbIndex = tt::CBIndex::c_10;                          // sum of inputs
constexpr auto kMeanBcastCbIndex = tt::CBIndex::c_12;                    // broadcasted mean
[[maybe_unused]] constexpr auto kXMinusMeanCbIndex = tt::CBIndex::c_13;  // (x - mean) - reserved for future use
constexpr auto kVarianceSumCbIndex = tt::CBIndex::c_14;                  // sum((x - mean)^2)
constexpr auto kVarianceReducedCbIndex = tt::CBIndex::c_15;              // variance (1/N * sum((x - mean)^2))
constexpr auto kRstdBcastCbIndex = tt::CBIndex::c_17;                    // broadcasted rstd
constexpr auto kXHatCbIndex = tt::CBIndex::c_18;                         // normalized x_hat
[[maybe_unused]] constexpr auto kOutputIntermediateCbIndex =
    tt::CBIndex::c_19;  // intermediate for output - reserved for future use

// CB sizes (some set to 2U for ping-pong)
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumEpsTiles = 1U;
constexpr uint32_t kNumMeanBcastTiles = 1U;
constexpr uint32_t kNumVarianceReducedTiles = 1U;
constexpr uint32_t kNumRstdBcastTiles = 1U;
constexpr uint32_t kNumSumTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kReturnMeanRstdDefineKey = "RETURN_MEAN_RSTD";

bool fits_in_l1_check(
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    const uint32_t float32_single_tile_size_bytes,
    tt::tt_metal::IDevice* device) {
    // Calculate available L1 memory
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Memory for input data CBs
    const uint64_t scaler_memory = kNumScalerTiles * bfloat16_single_tile_size_bytes;
    const uint64_t mask_memory = kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint64_t eps_memory = kNumEpsTiles * bfloat16_single_tile_size_bytes;
    const uint64_t gamma_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t beta_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t input_memory = Wt * bfloat16_single_tile_size_bytes;

    // Memory for output CBs
    const uint64_t output_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t mean_output_memory = kNumMeanBcastTiles * bfloat16_single_tile_size_bytes;
    const uint64_t rstd_output_memory = kNumRstdBcastTiles * bfloat16_single_tile_size_bytes;

    // Memory for intermediate computation CBs
    const uint64_t sum_memory = kNumSumTiles * float32_single_tile_size_bytes;
    const uint64_t mean_bcast_memory = kNumMeanBcastTiles * bfloat16_single_tile_size_bytes;
    const uint64_t variance_sum_memory = kNumSumTiles * float32_single_tile_size_bytes;
    const uint64_t variance_reduced_memory = kNumVarianceReducedTiles * bfloat16_single_tile_size_bytes;
    const uint64_t rstd_bcast_memory = kNumRstdBcastTiles * bfloat16_single_tile_size_bytes;
    const uint64_t x_hat_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t output_intermediate_memory = block_size * bfloat16_single_tile_size_bytes;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes =
        scaler_memory + mask_memory + eps_memory + gamma_memory + beta_memory + input_memory + output_memory +
        mean_output_memory + rstd_output_memory + sum_memory + mean_bcast_memory + variance_sum_memory +
        variance_reduced_memory + rstd_bcast_memory + x_hat_memory + output_intermediate_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

}  // namespace

namespace ttml::metal::ops::layernorm_fw::device {

struct LayerNormForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const LayerNormForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* gamma_buffer,
    const tt::tt_metal::Buffer* beta_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* mean_buffer,
    const tt::tt_metal::Buffer* rstd_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    bool return_mean_rstd) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many rows this core will process
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core {} not in specified core ranges", core.str());
        }

        // Reader kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(),
             gamma_buffer->address(),
             beta_buffer->address(),
             num_rows_written,
             num_rows_per_core});

        // Writer kernel runtime args
        std::vector<uint32_t> writer_args = {
            output_buffer->address(),
            mean_buffer ? mean_buffer->address() : 0,
            rstd_buffer ? rstd_buffer->address() : 0,
            num_rows_written,
            num_rows_per_core};

        SetRuntimeArgs(program, kernels.writer, core, writer_args);

        num_rows_written += num_rows_per_core;
    }
}

LayerNormForwardProgramFactory::cached_program_t LayerNormForwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const float epsilon = operation_attributes.epsilon;
    const bool return_mean_rstd = operation_attributes.return_mean_rstd;

    // Check input shape is [B, N, S, C]
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);

    // Check gamma shape is [1, 1, 1, C]
    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(gamma_shape.rank() == 4, "Gamma tensor must be 4D [1, 1, 1, C], got shape {}", gamma_shape);

    // Check beta shape is [1, 1, 1, C]
    const auto& beta_shape = beta.logical_shape();
    TT_FATAL(beta_shape.rank() == 4, "Beta tensor must be 4D [1, 1, 1, C], got shape {}", beta_shape);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(input_data_format);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // -------------------------------------------------------------------------
    // 1) Precompute tensor dimensions and padding
    // -------------------------------------------------------------------------
    const auto& padded_tensor_shape = input.padded_shape();
    uint32_t padded_tensor_volume = padded_tensor_shape.volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t total_rows_to_process =
        (padded_tensor_shape[-2] * padded_tensor_shape[-3] * padded_tensor_shape[-4]) / tt::constants::TILE_HEIGHT;

    // Get the number of inner dimension (assumes divisible by TILE_WIDTH)
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    uint32_t block_size = get_block_size(Wt, 2U);  // Need 2 extra registers for layernorm forward

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));
    uint32_t packed_eps = pack_two_bfloat16_to_uint32(epsilon);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const bool everything_fits_in_l1 =
        fits_in_l1_check(Wt, block_size, bfloat16_single_tile_size_bytes, float32_single_tile_size_bytes, device);

    const uint32_t num_input_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_gamma_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_beta_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_x_hat_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_output_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;

    auto data_format = input_data_format;

    // Input data CBs
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    [[maybe_unused]] auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);
    [[maybe_unused]] auto cb_eps = create_circular_buffer(
        program, all_cores, kEpsCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumEpsTiles);
    [[maybe_unused]] auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_gamma_tiles);
    [[maybe_unused]] auto cb_beta = create_circular_buffer(
        program, all_cores, kBetaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_beta_tiles);
    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    // Output CBs
    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_output_tiles);
    [[maybe_unused]] auto cb_mean = create_circular_buffer(
        program, all_cores, kMeanCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMeanBcastTiles);
    [[maybe_unused]] auto cb_rstd = create_circular_buffer(
        program, all_cores, kRstdCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRstdBcastTiles);

    // Intermediate computation CBs
    [[maybe_unused]] auto cb_sum = create_circular_buffer(
        program, all_cores, kSumCbIndex, data_format, float32_single_tile_size_bytes, kNumSumTiles);
    [[maybe_unused]] auto cb_mean_bcast = create_circular_buffer(
        program, all_cores, kMeanBcastCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMeanBcastTiles);
    [[maybe_unused]] auto cb_variance_sum = create_circular_buffer(
        program, all_cores, kVarianceSumCbIndex, data_format, float32_single_tile_size_bytes, kNumSumTiles);
    [[maybe_unused]] auto cb_variance_reduced = create_circular_buffer(
        program,
        all_cores,
        kVarianceReducedCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumVarianceReducedTiles);
    [[maybe_unused]] auto cb_rstd_bcast = create_circular_buffer(
        program, all_cores, kRstdBcastCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRstdBcastTiles);
    [[maybe_unused]] auto cb_x_hat = create_circular_buffer(
        program, all_cores, kXHatCbIndex, data_format, bfloat16_single_tile_size_bytes, num_x_hat_tiles);
    // Intermediate CB for output computation (x_hat * gamma before adding beta)
    [[maybe_unused]] auto cb_output_intermediate = create_circular_buffer(
        program, all_cores, kOutputIntermediateCbIndex, data_format, bfloat16_single_tile_size_bytes, block_size);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* gamma_buffer = gamma.buffer();
    TT_FATAL(
        gamma_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Gamma buffer must be in DRAM. Gamma buffer of type {}",
        enchantum::to_string(gamma_buffer->buffer_type()));

    auto* beta_buffer = beta.buffer();
    TT_FATAL(
        beta_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Beta buffer must be in DRAM. Beta buffer of type {}",
        enchantum::to_string(beta_buffer->buffer_type()));

    auto* output_buffer = output[0].value().buffer();
    TT_FATAL(
        output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    tt::tt_metal::Buffer* mean_buffer = nullptr;
    tt::tt_metal::Buffer* rstd_buffer = nullptr;
    if (return_mean_rstd) {
        mean_buffer = output[1].value().buffer();
        TT_FATAL(
            mean_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Mean buffer must be in DRAM. Mean buffer of type {}",
            enchantum::to_string(mean_buffer->buffer_type()));

        rstd_buffer = output[2].value().buffer();
        TT_FATAL(
            rstd_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Rstd buffer must be in DRAM. Rstd buffer of type {}",
            enchantum::to_string(rstd_buffer->buffer_type()));
    }

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }
    if (return_mean_rstd) {
        defines[kReturnMeanRstdDefineKey] = "1";
    }

    LayerNormForwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{packed_scaler, packed_eps, block_size, mask_w, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    if (return_mean_rstd) {
        tt::tt_metal::TensorAccessorArgs(mean_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(rstd_buffer).append_to(writer_compile_time_args);
    } else {
        // Add dummy TensorAccessorArgs for mean and rstd (they won't be used)
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    }
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        mask_w,                     // mask_w
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            mask_w,                     // mask_w
            Wt                          // num_inner / TILE_W
        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        gamma_buffer,
        beta_buffer,
        output_buffer,
        mean_buffer,
        rstd_buffer,
        num_cores,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2,
        return_mean_rstd);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* layernorm_fw_reader_kernel_id  = */ kernels.reader,
         /* layernorm_fw_writer_kernel_id  = */ kernels.writer,
         /* layernorm_fw_kernel_group_1_id = */ kernels.compute_group_1,
         /* layernorm_fw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void LayerNormForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& layernorm_fw_reader_kernel_id = shared_variables.layernorm_fw_reader_kernel_id;
    auto& layernorm_fw_writer_kernel_id = shared_variables.layernorm_fw_writer_kernel_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    auto* input_buffer = tensor_args.input.buffer();
    auto* gamma_buffer = tensor_args.gamma.buffer();
    auto* beta_buffer = tensor_args.beta.buffer();

    auto* output_buffer = output[0].value().buffer();
    tt::tt_metal::Buffer* mean_buffer = nullptr;
    tt::tt_metal::Buffer* rstd_buffer = nullptr;
    if (operation_attributes.return_mean_rstd) {
        mean_buffer = output[1].value().buffer();
        rstd_buffer = output[2].value().buffer();
    }

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, layernorm_fw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, layernorm_fw_writer_kernel_id);

    // Process core_group_1 - iterate over actual cores
    for (const auto& core_range : core_group_1.ranges()) {
        for (auto core : tt::tt_metal::CoreRange(core_range)) {
            // Update input buffers for the reader kernel
            {
                auto& runtime_args = reader_runtime_args[core.x][core.y];
                runtime_args[kInputBufferIdx] = input_buffer->address();
                runtime_args[kGammaBufferIdx] = gamma_buffer->address();
                runtime_args[kBetaBufferIdx] = beta_buffer->address();
            }

            // Update output buffers for the writer kernel
            {
                auto& runtime_args = writer_runtime_args[core.x][core.y];
                runtime_args[kOutputBufferIdx] = output_buffer->address();
                if (operation_attributes.return_mean_rstd) {
                    runtime_args[kMeanBufferIdx] = mean_buffer->address();
                    runtime_args[kRstdBufferIdx] = rstd_buffer->address();
                }
            }
        }
    }

    // Process core_group_2 - iterate over actual cores
    for (const auto& core_range : core_group_2.ranges()) {
        for (auto core : tt::tt_metal::CoreRange(core_range)) {
            // Update input buffers for the reader kernel
            {
                auto& runtime_args = reader_runtime_args[core.x][core.y];
                runtime_args[kInputBufferIdx] = input_buffer->address();
                runtime_args[kGammaBufferIdx] = gamma_buffer->address();
                runtime_args[kBetaBufferIdx] = beta_buffer->address();
            }

            // Update output buffers for the writer kernel
            {
                auto& runtime_args = writer_runtime_args[core.x][core.y];
                runtime_args[kOutputBufferIdx] = output_buffer->address();
                if (operation_attributes.return_mean_rstd) {
                    runtime_args[kMeanBufferIdx] = mean_buffer->address();
                    runtime_args[kRstdBufferIdx] = rstd_buffer->address();
                }
            }
        }
    }
}

}  // namespace ttml::metal::ops::layernorm_fw::device
