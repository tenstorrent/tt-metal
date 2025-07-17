// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_program_factory.hpp"

#include <bit>
#include <cstdint>

#include "metal/ops/common/program_utils.hpp"
#include "rmsnorm_fw_device_operation_types.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/dataflow/"
    "writer_rmsnorm_fw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/dataflow/"
    "reader_rmsnorm_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/compute/rmsnorm_fw_kernel.cpp";

// reader runtime args
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kGammaBufferIdx = 1U;
// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;
constexpr uint32_t kRMSOutputBufferIdx = 1U;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskCbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kEpsCbIndex = tt::CBIndex::c_3;
constexpr auto kGammaCbIndex = tt::CBIndex::c_4;
constexpr auto kRmsBeforeReductionCbIndex = tt::CBIndex::c_5;
constexpr auto kRmsAfterReductionCbIndex = tt::CBIndex::c_6;
constexpr auto kInverseRmsAfterReductionCbIndex = tt::CBIndex::c_7;
constexpr auto kOutputCbIndex = tt::CBIndex::c_8;
constexpr auto kRmsOutputCbIndex = tt::CBIndex::c_9;
constexpr auto kOutputIntermediateCbIndex = tt::CBIndex::c_10;

constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsBeforeReductionTiles = 2U;
constexpr uint32_t kNumRmsAfterReductionTiles = 2U;
constexpr uint32_t kNumInverseRmsAfterReductionTiles = 2U;
constexpr uint32_t kNumRmsOutputTiles = 2U;
constexpr uint32_t kNumEpsTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kReturnRMSDefineKey = "RETURN_RMS";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kEverythingExceptGammaFitsInL1DefineKey = "EVERYTHING_EXCEPT_GAMMA_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::rmsnorm_fw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct RMSNormForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
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

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

/**
 * Create a compute kernel with the given compile-time arguments.
 */
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

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const RMSNormForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* gamma_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* rms_output_buffer,
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

        // Reader kernel: (input_addr, gamma_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(), gamma_buffer->address(), num_rows_per_core, num_rows_written});

        // Writer kernel: (dst_addr, dst_rms_addr number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(), rms_output_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

RMSNormForwardProgramFactory::cached_program_t RMSNormForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;

    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

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

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_inner = input.logical_shape()[-1];

    // compile arguments
    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(1.F / static_cast<float>(num_inner));
    uint32_t packed_eps = pack_two_bfloat16_to_uint32(args.epsilon);
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;
    uint32_t block_size = get_block_size(Wt, 3U);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    uint32_t twice_block_size = 2U * block_size;
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Memory allocation for L1 cache
    const uint64_t weight_memory = 2U * Wt * bfloat16_single_tile_size_bytes;
    const uint64_t mask_scaler_eps_memory =
        (kNumMaskTiles + kNumScalerTiles + kNumEpsTiles) * bfloat16_single_tile_size_bytes;
    const uint64_t rms_memory =
        (kNumRmsBeforeReductionTiles + kNumRmsAfterReductionTiles + kNumInverseRmsAfterReductionTiles) *
        float32_single_tile_size_bytes;
    const uint64_t block_memory = 2 * twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t rms_output_memory = kNumRmsOutputTiles * bfloat16_single_tile_size_bytes;
    // Total L1 memory required
    const uint64_t required_L1_in_bytes =
        weight_memory + mask_scaler_eps_memory + rms_memory + block_memory + rms_output_memory;
    const bool everything_fits_in_l1 = required_L1_in_bytes <= available_L1_in_bytes;

    const uint64_t required_L1_in_bytes_except_gamma = required_L1_in_bytes - Wt * bfloat16_single_tile_size_bytes +
                                                       twice_block_size * bfloat16_single_tile_size_bytes;
    const bool everything_except_gamma_fits_in_l1 = required_L1_in_bytes_except_gamma <= available_L1_in_bytes;

    const uint32_t num_input_tiles =
        (everything_fits_in_l1 | everything_except_gamma_fits_in_l1) ? Wt : twice_block_size;
    const uint32_t num_gamma_tiles = everything_fits_in_l1 ? Wt : twice_block_size;

    auto data_format = input_data_format;
    auto precise_data_format = tt::DataFormat::Float32;

    auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    auto cb_mask = create_circular_buffer(
        program, all_cores, kMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);

    auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_eps = create_circular_buffer(
        program, all_cores, kEpsCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumEpsTiles);

    auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_gamma_tiles);

    auto cb_rms_before_reduction_intermediate = create_circular_buffer(
        program,
        all_cores,
        kRmsBeforeReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumRmsBeforeReductionTiles);

    auto cb_rms_after_reduction_intermediate = create_circular_buffer(
        program,
        all_cores,
        kRmsAfterReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumRmsAfterReductionTiles);

    auto cb_inverse_rms_after_reduction_intermediate = create_circular_buffer(
        program,
        all_cores,
        kInverseRmsAfterReductionCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kNumInverseRmsAfterReductionTiles);

    auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

    auto cb_rms_output = create_circular_buffer(
        program, all_cores, kRmsOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsOutputTiles);

    auto cb_output_intermediate = create_circular_buffer(
        program, all_cores, kOutputIntermediateCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* gamma_buffer = gamma.buffer();
    TT_FATAL(
        gamma_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Gamma buffer must be in DRAM. Gamma buffer of type {}",
        enchantum::to_string(gamma_buffer->buffer_type()));

    auto* output_buffer = output.front().buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    auto* rms_output_buffer = output.back().buffer();
    TT_FATAL(
        rms_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "RMS output buffer must be in DRAM. RMS output buffer of type {}",
        enchantum::to_string(rms_output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    if (everything_except_gamma_fits_in_l1) {
        defines[kEverythingExceptGammaFitsInL1DefineKey] = "1";
    }

    if (args.return_intermediates) {
        defines[kReturnRMSDefineKey] = "1";
    }

    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    RMSNormForwardKernels kernels;
    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ {packed_scaler, packed_eps, mask_w, Wt, block_size},
        defines,
        kReaderKernelPath);
    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ {Wt, block_size}, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for rmsnorm_fw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
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
        output_buffer,
        rms_output_buffer,
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

void RMSNormForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& rmsnorm_fw_reader_kernel = shared_vars.rmsnorm_fw_reader_kernel_id;
    auto& rmsnorm_fw_writer_kernel = shared_vars.rmsnorm_fw_writer_kernel_id;
    auto& rmsnorm_fw_group_1_kernel = shared_vars.rmsnorm_fw_kernel_group_1_id;
    auto& rmsnorm_fw_group_2_kernel = shared_vars.rmsnorm_fw_kernel_group_2_id;
    auto& core_group_1 = shared_vars.core_group_1;
    auto& core_group_2 = shared_vars.core_group_2;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    auto* input_buffer = input_tensor.buffer();
    auto* gamma_buffer = gamma_tensor.buffer();

    const auto& output_tensor = output.front();
    const auto& rms_output_tensor = output.back();
    auto* output_buffer = output_tensor.buffer();
    auto* rms_output_buffer = rms_output_tensor.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, rmsnorm_fw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, rmsnorm_fw_writer_kernel);
    auto& group_1_runtime_args = GetRuntimeArgs(program, rmsnorm_fw_group_1_kernel);
    // we need to initialize it with something, but if group 2 is  empty it will be used in the loop
    auto& group_2_runtime_args =
        core_group_2.ranges().empty() ? group_1_runtime_args : GetRuntimeArgs(program, rmsnorm_fw_group_2_kernel);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kGammaBufferIdx] = gamma_buffer->address();
        }
        // Update destination buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
            runtime_args[kRMSOutputBufferIdx] = rms_output_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::rmsnorm_fw::device
