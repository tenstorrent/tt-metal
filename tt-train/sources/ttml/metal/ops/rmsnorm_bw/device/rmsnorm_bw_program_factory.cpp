// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"
#include <enchantum/enchantum.hpp>

#include <cstdint>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/writer_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/reader_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/compute/rmsnorm_bw_kernel.cpp";

// Buffer indices
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kGammaBufferIdx = 1U;
constexpr uint32_t kRmsBufferIdx = 2U;
constexpr uint32_t kDLdoutBufferIdx = 3U;

// Writer buffer indices
constexpr uint32_t kDaBufferIdx = 0;
constexpr uint32_t kDgammaComponentsBufferIdx = 1U;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;
constexpr auto kRmsACbIndex = tt::CBIndex::c_4;
constexpr auto kDLoutCbIndex = tt::CBIndex::c_5;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_7;
constexpr auto kDLdgammaComponentsCbIndex = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr auto kRecipRmsACbIndex = tt::CBIndex::c_9;
constexpr auto kScaleCbIndex = tt::CBIndex::c_10;
constexpr auto kScaleBcastedCbIndex = tt::CBIndex::c_11;

// Some of the below constants are set to 2U because we might need to push a new value before poping the old one.
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsATiles = 2U;
constexpr uint32_t kNumMatMulReduceTiles = 1U;
constexpr uint32_t kNumRecipRmsATiles = 1U;
constexpr uint32_t kNumScaleTiles = 2U;
constexpr uint32_t kNumScaleBcastedTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compile_time_args,
            .defines = defines});
}

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const RMSNormBackwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* gamma_buffer,
    const tt::tt_metal::Buffer* rms_buffer,
    const tt::tt_metal::Buffer* dLdout_buffer,
    const tt::tt_metal::Buffer* da_buffer,
    const tt::tt_metal::Buffer* dgamma_buffer,
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

        // Reader kernel: (input_addr, gamma_addr, rms_addr, dLdout_addr, num_rows, offset)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(),
             gamma_buffer->address(),
             rms_buffer->address(),
             dLdout_buffer->address(),
             num_rows_per_core,
             num_rows_written});

        // Writer kernel: (da_addr, dgamma_addr, num_rows, offset)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {da_buffer->address(), dgamma_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

bool fits_in_l1_check(
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    const uint32_t float32_single_tile_size_bytes,
    ttnn::IDevice* device) {
    const uint32_t twice_block_size = 2U * block_size;

    // Move the memory check to a separate function. And just return boolean whether it fits in L1 or not.
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Memory of input tensors
    const uint64_t input_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t mask_memory = kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint64_t scaler_memory = kNumScalerTiles * bfloat16_single_tile_size_bytes;
    const uint64_t gamma_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t rms_a_memory = kNumRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dout_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t matmul_reduce_memory = kNumMatMulReduceTiles * bfloat16_single_tile_size_bytes;
    // Memory for output tensors
    const uint64_t dL_da_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dgamma_components_memory = Wt * bfloat16_single_tile_size_bytes;
    // Memory for intermediate computations
    const uint64_t recip_rms_a_bcasted_memory = kNumRecipRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t scale_memory = kNumScaleTiles * float32_single_tile_size_bytes;
    const uint64_t scale_bcasted_memory = kNumScaleBcastedTiles * float32_single_tile_size_bytes;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes = input_memory + mask_memory + scaler_memory + gamma_memory + rms_a_memory +
                                          dL_dout_memory + matmul_reduce_memory + dL_da_memory +
                                          dL_dgamma_components_memory + recip_rms_a_bcasted_memory + scale_memory +
                                          scale_bcasted_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

RMSNormBackwardProgramFactory::cached_program_t RMSNormBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& rms = tensor_args.rms;
    const auto& dLdout = tensor_args.dL_dout;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // Get the number of inner dimension
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    uint32_t block_size = get_block_size(Wt, 2U);  // We need two extra registers during calculation

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const bool everything_fits_in_l1 =
        fits_in_l1_check(Wt, block_size, bfloat16_single_tile_size_bytes, float32_single_tile_size_bytes, device);

    const uint32_t num_input_tiles =
        (everything_fits_in_l1) ? Wt : twice_block_size;  // If everything fits in L1, read Wt tiles, else read 2x block

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;

    auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);
    auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    auto cb_rms_a = create_circular_buffer(
        program, all_cores, kRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsATiles);
    auto cb_dLdout = create_circular_buffer(
        program, all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMatMulReduceTiles);
    auto cb_dL_da = create_circular_buffer(
        program, all_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    auto cb_dL_dgamma_components = create_circular_buffer(
        program, all_cores, kDLdgammaComponentsCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    auto cb_recip_rms_a_bcasted = create_circular_buffer(
        program, all_cores, kRecipRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRecipRmsATiles);
    auto cb_scale = create_circular_buffer(
        program, all_cores, kScaleCbIndex, precise_data_format, float32_single_tile_size_bytes, kNumScaleTiles);
    auto cb_scale_bcasted = create_circular_buffer(
        program,
        all_cores,
        kScaleBcastedCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumScaleBcastedTiles);

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

    auto* rms_buffer = rms.buffer();
    TT_FATAL(
        rms_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "RMS buffer must be in DRAM. RMS buffer of type {}",
        enchantum::to_string(rms_buffer->buffer_type()));

    auto* dLdout_buffer = dLdout.buffer();
    TT_FATAL(
        dLdout_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dLdout_buffer->buffer_type()));

    auto* dL_da_buffer = output[0].buffer();
    TT_FATAL(
        dL_da_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_da buffer must be in DRAM. dL_da buffer of type {}",
        enchantum::to_string(dL_da_buffer->buffer_type()));

    auto* dL_dgamma_components_buffer = output[1].buffer();
    TT_FATAL(
        dL_dgamma_components_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "dL_dgamma buffer must be in DRAM. dL_dgamma buffer of type {}",
        enchantum::to_string(dL_dgamma_components_buffer->buffer_type()));

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    RMSNormBackwardKernels kernels;
    kernels.reader =
        create_reader_kernel(program, all_cores, {packed_scaler, block_size, mask_w, Wt}, defines, kReaderKernelPath);

    kernels.writer = create_writer_kernel(
        program,
        all_cores,
        {
            block_size,
            Wt,
        },
        defines,
        kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for cross_entropy_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        mask_w,                     // mask_w
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            mask_w,                     // mask_w
            Wt                          // num_inner / TILE_W
        };

        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath);
    }
    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        gamma_buffer,
        rms_buffer,
        dLdout_buffer,
        dL_da_buffer,
        dL_dgamma_components_buffer,
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
        {/* rmsnorm_fw_reader_kernel_id  = */ kernels.reader,
         /* rmsnorm_fw_writer_kernel_id  = */ kernels.writer,
         /* rmsnorm_fw_kernel_group_1_id = */ kernels.compute_group_1,
         /* rmsnorm_fw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void RMSNormBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& rmsnorm_bw_reader_kernel_id = shared_variables.rmsnorm_bw_reader_kernel_id;
    auto& rmsnorm_bw_writer_kernel_id = shared_variables.rmsnorm_bw_writer_kernel_id;
    auto& rmsnorm_bw_kernel_group_1_id = shared_variables.rmsnorm_bw_kernel_group_1_id;
    auto& rmsnorm_bw_kernel_group_2_id = shared_variables.rmsnorm_bw_kernel_group_2_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* gamma_buffer = tensor_args.gamma.buffer();
    auto* rms_buffer = tensor_args.rms.buffer();
    auto* dLdout_buffer = tensor_args.dL_dout.buffer();

    auto* da_buffer = output[0].buffer();
    auto* dgamma_buffer = output[1].buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kGammaBufferIdx] = gamma_buffer->address();
            runtime_args[kRmsBufferIdx] = rms_buffer->address();
            runtime_args[kDLdoutBufferIdx] = dLdout_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kDaBufferIdx] = da_buffer->address();
            runtime_args[kDgammaComponentsBufferIdx] = dgamma_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device
