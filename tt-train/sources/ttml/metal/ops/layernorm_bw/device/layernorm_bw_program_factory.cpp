// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/dataflow/writer_layernorm_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/dataflow/reader_layernorm_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp";

// Reader buffer indices
constexpr uint32_t kGammaBufferIdx = 0;
constexpr uint32_t kXHatBufferIdx = 1U;
constexpr uint32_t kRstdBufferIdx = 2U;
constexpr uint32_t kDLdoutBufferIdx = 3U;

// Writer buffer indices
constexpr uint32_t kDxBufferIdx = 0;
constexpr uint32_t kDgammaComponentsBufferIdx = 1U;
constexpr uint32_t kDbetaComponentsBufferIdx = 2U;

// CBs with input data
constexpr auto kScalerCbIndex = tt::CBIndex::c_0;        // 1/N scaler
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;         // mask for width dimension
constexpr auto kGammaCbIndex = tt::CBIndex::c_2;         // gamma (scale parameter)
constexpr auto kXHatCbIndex = tt::CBIndex::c_3;          // x_hat (normalized input) from forward pass
constexpr auto kRstdCbIndex = tt::CBIndex::c_4;          // rstd from forward pass
constexpr auto kDLoutCbIndex = tt::CBIndex::c_5;         // upstream gradient
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;  // reduction vector

// CBs with output data
constexpr auto kDxCbIndex = tt::CBIndex::c_10;                // dx (input gradient)
constexpr auto kDgammaComponentsCbIndex = tt::CBIndex::c_11;  // dgamma components
constexpr auto kDbetaComponentsCbIndex = tt::CBIndex::c_12;   // dbeta components

// CBs with intermediate computations
constexpr auto kXNormalizedCbIndex = tt::CBIndex::c_13;       // x_normalized = (x - mean) * rstd
constexpr auto kDyGammaCbIndex = tt::CBIndex::c_14;           // dy * gamma
constexpr auto kDyGammaSumCbIndex = tt::CBIndex::c_15;        // sum(dy * gamma)
constexpr auto kDyGammaXnormSumCbIndex = tt::CBIndex::c_16;   // sum(dy * gamma * x_normalized)
constexpr auto kScaledDyGammaSumCbIndex = tt::CBIndex::c_17;  // (1/N) * sum(dy * gamma) - pre-scaled
constexpr auto kScaledDyGammaXnormSumCbIndex =
    tt::CBIndex::c_18;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled
constexpr auto kCbZeroIndex = tt::CBIndex::c_19;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled

// CB sizes (some set to 2U for ping-pong)
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumRstdTiles = 2U;
constexpr uint32_t kNumMatMulReduceTiles = 1U;
constexpr uint32_t kNumXNormalizedTiles = 2U;
constexpr uint32_t kNumDyGammaTiles = 2U;
constexpr uint32_t kNumDyGammaSumTiles = 1U;
constexpr uint32_t kNumDyGammaXnormSumTiles = 1U;
constexpr uint32_t kNumZeroTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

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
    const uint64_t gamma_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t x_hat_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t rstd_memory = kNumRstdTiles * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dout_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t matmul_reduce_memory = kNumMatMulReduceTiles * bfloat16_single_tile_size_bytes;

    // Memory for output CBs
    const uint64_t dx_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t dgamma_components_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t dbeta_components_memory = Wt * bfloat16_single_tile_size_bytes;

    // Memory for intermediate computation CBs
    const uint64_t x_normalized_memory = kNumXNormalizedTiles * bfloat16_single_tile_size_bytes;
    const uint64_t dy_gamma_memory = kNumDyGammaTiles * bfloat16_single_tile_size_bytes;
    const uint64_t dy_gamma_sum_memory = kNumDyGammaSumTiles * float32_single_tile_size_bytes;
    const uint64_t dy_gamma_xnorm_sum_memory = kNumDyGammaXnormSumTiles * float32_single_tile_size_bytes;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes = scaler_memory + mask_memory + gamma_memory + x_hat_memory + rstd_memory +
                                          dL_dout_memory + matmul_reduce_memory + dx_memory + dgamma_components_memory +
                                          dbeta_components_memory + x_normalized_memory + dy_gamma_memory +
                                          dy_gamma_sum_memory + dy_gamma_xnorm_sum_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

}  // namespace

namespace ttml::metal::ops::layernorm_bw::device {

struct LayerNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const LayerNormBackwardKernels& kernels,
    const tt::tt_metal::Buffer* gamma_buffer,
    const tt::tt_metal::Buffer* x_hat_buffer,
    const tt::tt_metal::Buffer* rstd_buffer,
    const tt::tt_metal::Buffer* dLdout_buffer,
    const tt::tt_metal::Buffer* dx_buffer,
    const tt::tt_metal::Buffer* dgamma_components_buffer,
    const tt::tt_metal::Buffer* dbeta_components_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2) {
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
            {gamma_buffer->address(),
             x_hat_buffer->address(),
             rstd_buffer->address(),
             dLdout_buffer->address(),
             num_rows_written,
             num_rows_per_core});

        // Writer kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {dx_buffer->address(),
             dgamma_components_buffer->address(),
             dbeta_components_buffer->address(),
             num_rows_written,
             num_rows_per_core});

        num_rows_written += num_rows_per_core;
    }
}

LayerNormBackwardProgramFactory::cached_program_t LayerNormBackwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& x_hat = tensor_args.x_hat;
    const auto& rstd = tensor_args.rstd;
    const auto& dLdout = tensor_args.dL_dout;

    // Check input shape is [B, 1, S, C]
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, 1, C], got shape {}", input_shape);

    // Check gamma shape is [1, 1, 1, C]
    const auto& gamma_shape = gamma.logical_shape();
    std::cout << "Gamma shape: " << gamma_shape << std::endl;
    TT_FATAL(gamma_shape.rank() == 4, "Gamma tensor must be 4D [1, 1, 1, C], got shape {}", gamma_shape);

    // Check x_hat shape is [B, 1, S, C] - same as input
    const auto& x_hat_shape = x_hat.logical_shape();
    TT_FATAL(x_hat_shape.rank() == 4, "X_hat tensor must be 4D [B, N, S, C], got shape {}", x_hat_shape);

    // Check rstd shape is [B, 1, S, 1]
    // Note: LayerNorm computes statistics across N and C dimensions, so dimension 1 is 1
    const auto& rstd_shape = rstd.logical_shape();
    TT_FATAL(rstd_shape.rank() == 4, "Rstd tensor must be 4D [B, 1, S, 1], got shape {}", rstd_shape);

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
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // Get the number of inner dimension (assumes divisible by TILE_WIDTH)
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    uint32_t block_size = get_block_size(Wt, 2U);  // Need 2 extra registers for layernorm backward

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const bool everything_fits_in_l1 =
        fits_in_l1_check(Wt, block_size, bfloat16_single_tile_size_bytes, float32_single_tile_size_bytes, device);

    std::cout << "Everything fits in L1: " << everything_fits_in_l1 << std::endl;

    const uint32_t num_input_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;

    const uint32_t num_x_hat_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;

    auto data_format = input_data_format;
    // auto precise_data_format = tt::DataFormat::Float32;

    // Input data CBs
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    [[maybe_unused]] auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);
    [[maybe_unused]] auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_x_hat = create_circular_buffer(
        program, all_cores, kXHatCbIndex, data_format, bfloat16_single_tile_size_bytes, num_x_hat_tiles);
    [[maybe_unused]] auto cb_rstd = create_circular_buffer(
        program, all_cores, kRstdCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRstdTiles);
    [[maybe_unused]] auto cb_dLdout = create_circular_buffer(
        program, all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMatMulReduceTiles);

    // Output CBs
    [[maybe_unused]] auto cb_dx = create_circular_buffer(
        program, all_cores, kDxCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_dgamma_components = create_circular_buffer(
        program, all_cores, kDgammaComponentsCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_dbeta_components = create_circular_buffer(
        program, all_cores, kDbetaComponentsCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    // Intermediate computation CBs
    [[maybe_unused]] auto cb_x_normalized = create_circular_buffer(
        program, all_cores, kXNormalizedCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumXNormalizedTiles);
    [[maybe_unused]] auto cb_dy_gamma = create_circular_buffer(
        program, all_cores, kDyGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumDyGammaTiles);
    [[maybe_unused]] auto cb_dy_gamma_sum = create_circular_buffer(
        program, all_cores, kDyGammaSumCbIndex, data_format, float32_single_tile_size_bytes, kNumDyGammaSumTiles);
    [[maybe_unused]] auto cb_dy_gamma_xnorm_sum = create_circular_buffer(
        program,
        all_cores,
        kDyGammaXnormSumCbIndex,
        data_format,
        float32_single_tile_size_bytes,
        kNumDyGammaXnormSumTiles);
    [[maybe_unused]] auto cb_scaled_dy_gamma_sum = create_circular_buffer(
        program, all_cores, kScaledDyGammaSumCbIndex, data_format, float32_single_tile_size_bytes, kNumDyGammaSumTiles);
    [[maybe_unused]] auto cb_scaled_dy_gamma_xnorm_sum = create_circular_buffer(
        program,
        all_cores,
        kScaledDyGammaXnormSumCbIndex,
        data_format,
        float32_single_tile_size_bytes,
        kNumDyGammaXnormSumTiles);
    [[maybe_unused]] auto cb_zero = create_circular_buffer(
        program, all_cores, kCbZeroIndex, data_format, bfloat16_single_tile_size_bytes, kNumZeroTiles);
    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* gamma_buffer = gamma.buffer();
    TT_FATAL(
        gamma_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Gamma buffer must be in DRAM. Gamma buffer of type {}",
        enchantum::to_string(gamma_buffer->buffer_type()));

    auto* x_hat_buffer = x_hat.buffer();
    TT_FATAL(
        x_hat_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "X_hat buffer must be in DRAM. X_hat buffer of type {}",
        enchantum::to_string(x_hat_buffer->buffer_type()));

    auto* rstd_buffer = rstd.buffer();
    TT_FATAL(
        rstd_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Rstd buffer must be in DRAM. Rstd buffer of type {}",
        enchantum::to_string(rstd_buffer->buffer_type()));

    auto* dLdout_buffer = dLdout.buffer();
    TT_FATAL(
        dLdout_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dLdout_buffer->buffer_type()));

    auto* dx_buffer = output[0].buffer();
    TT_FATAL(
        dx_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dx buffer must be in DRAM. dx buffer of type {}",
        enchantum::to_string(dx_buffer->buffer_type()));

    auto* dgamma_components_buffer = output[1].buffer();
    TT_FATAL(
        dgamma_components_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dgamma_components buffer must be in DRAM. dgamma_components buffer of type {}",
        enchantum::to_string(dgamma_components_buffer->buffer_type()));

    auto* dbeta_components_buffer = output[2].buffer();
    TT_FATAL(
        dbeta_components_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dbeta_components buffer must be in DRAM. dbeta_components buffer of type {}",
        enchantum::to_string(dbeta_components_buffer->buffer_type()));

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    LayerNormBackwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{packed_scaler, block_size, mask_w, Wt};
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(x_hat_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(rstd_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dx_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dgamma_components_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dbeta_components_buffer).append_to(writer_compile_time_args);
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
        gamma_buffer,
        x_hat_buffer,
        rstd_buffer,
        dLdout_buffer,
        dx_buffer,
        dgamma_components_buffer,
        dbeta_components_buffer,
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
        {/* layernorm_bw_reader_kernel_id  = */ kernels.reader,
         /* layernorm_bw_writer_kernel_id  = */ kernels.writer,
         /* layernorm_bw_kernel_group_1_id = */ kernels.compute_group_1,
         /* layernorm_bw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void LayerNormBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& layernorm_bw_reader_kernel_id = shared_variables.layernorm_bw_reader_kernel_id;
    auto& layernorm_bw_writer_kernel_id = shared_variables.layernorm_bw_writer_kernel_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    auto* gamma_buffer = tensor_args.gamma.buffer();
    auto* x_hat_buffer = tensor_args.x_hat.buffer();
    auto* rstd_buffer = tensor_args.rstd.buffer();
    auto* dLdout_buffer = tensor_args.dL_dout.buffer();

    auto* dx_buffer = output[0].buffer();
    auto* dgamma_buffer = output[1].buffer();
    auto* dbeta_buffer = output[2].buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, layernorm_bw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, layernorm_bw_writer_kernel_id);

    // Process core_group_1 - iterate over actual cores
    for (const auto& core_range : core_group_1.ranges()) {
        for (auto core : tt::tt_metal::CoreRange(core_range)) {
            // Update input buffers for the reader kernel
            {
                auto& runtime_args = reader_runtime_args[core.x][core.y];
                runtime_args[kGammaBufferIdx] = gamma_buffer->address();
                runtime_args[kXHatBufferIdx] = x_hat_buffer->address();
                runtime_args[kRstdBufferIdx] = rstd_buffer->address();
                runtime_args[kDLdoutBufferIdx] = dLdout_buffer->address();
            }

            // Update output buffers for the writer kernel
            {
                auto& runtime_args = writer_runtime_args[core.x][core.y];
                runtime_args[kDxBufferIdx] = dx_buffer->address();
                runtime_args[kDgammaComponentsBufferIdx] = dgamma_buffer->address();
                runtime_args[kDbetaComponentsBufferIdx] = dbeta_buffer->address();
            }
        }
    }

    // Process core_group_2 - iterate over actual cores
    for (const auto& core_range : core_group_2.ranges()) {
        for (auto core : tt::tt_metal::CoreRange(core_range)) {
            // Update input buffers for the reader kernel
            {
                auto& runtime_args = reader_runtime_args[core.x][core.y];
                runtime_args[kGammaBufferIdx] = gamma_buffer->address();
                runtime_args[kXHatBufferIdx] = x_hat_buffer->address();
                runtime_args[kRstdBufferIdx] = rstd_buffer->address();
                runtime_args[kDLdoutBufferIdx] = dLdout_buffer->address();
            }

            // Update output buffers for the writer kernel
            {
                auto& runtime_args = writer_runtime_args[core.x][core.y];
                runtime_args[kDxBufferIdx] = dx_buffer->address();
                runtime_args[kDgammaComponentsBufferIdx] = dgamma_buffer->address();
                runtime_args[kDbetaComponentsBufferIdx] = dbeta_buffer->address();
            }
        }
    }
}

}  // namespace ttml::metal::ops::layernorm_bw::device
