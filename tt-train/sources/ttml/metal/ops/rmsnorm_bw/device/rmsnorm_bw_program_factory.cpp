// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/writer_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/dataflow/reader_rmsnorm_bw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/compute/rmsnorm_bw_kernel.cpp";

// Buffer indices (adjust as needed)
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kGammaBufferIdx = 1U;
constexpr uint32_t kRmsBufferIdx = 2U;
constexpr uint32_t kDLdoutBufferIdx = 3U;

// Writer buffer indices
constexpr uint32_t kDxBufferIdx = 0;
constexpr uint32_t kDgammaBufferIdx = 1U;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;  // Unused atm
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;  // Number of activations, i.e. c in the paper
constexpr auto kRmsACbIndex = tt::CBIndex::c_4;
constexpr auto kDLoutCbIndex = tt::CBIndex::c_5;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_6;
constexpr auto kDLdgammaCbIndex = tt::CBIndex::c_7;
// CBs with intermediate computations
constexpr auto kScaledGainCbIndex = tt::CBIndex::c_8;
constexpr auto kGainedDLdoutCbIndex = tt::CBIndex::c_9;
constexpr auto kScaleCbIndex = tt::CBIndex::c_10;
constexpr auto kMsACbIndex = tt::CBIndex::c_11;
constexpr auto kCByMsACbIndex = tt::CBIndex::c_12;
constexpr auto kRhsCbIndex = tt::CBIndex::c_13;
constexpr auto kAOverRmsACbIndex = tt::CBIndex::c_14;
constexpr auto kDLdgammaComponentsCbIndex = tt::CBIndex::c_15;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

uint32_t pack_two_bfloat16_to_uint32(float value) {
    uint32_t uint32_data = std::bit_cast<uint32_t>(value);
    uint32_t casted_uint16_data = uint32_data >> 16U;
    return casted_uint16_data | (casted_uint16_data << 16);
}

uint32_t get_block_size(uint32_t num_inner) {
    const uint32_t max_block_size = 4U;  // 4 is the maximum block size for enabled fp32 dest acc
    for (uint32_t block_size = max_block_size; block_size > 1; block_size--) {
        if (num_inner % block_size == 0) {
            return block_size;
        }
    }
    return 1U;
}

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
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
    uint32_t num_rows_per_core,
    const tt::tt_metal::CoreRangeSet& all_cores) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

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

        // Writer kernel: (dx_addr, dgamma_addr, num_rows, offset)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {da_buffer->address(), dgamma_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

RMSNormBackwardProgramFactory::cached_program_t RMSNormBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // 1) Setup device, data formats, tile sizes, and compute split
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& rms = tensor_args.rms;
    const auto& dLdout = tensor_args.dL_dout;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size_bytes = tt::tt_metal::detail::TileSize(data_format);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.padded_volume();
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

    // For simplicity, use 1 row per core (can be improved)
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t num_rows_per_core = (total_rows_to_process + num_cores - 1) / num_cores;

    CoreCoord core_start{0, 0};
    CoreCoord core_end{num_cores_x - 1, num_cores_y - 1};
    tt::tt_metal::CoreRangeSet all_cores(
        std::vector<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange(core_start, core_end)});

    // compile arguments
    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(1.F / static_cast<float>(num_inner));
    uint32_t mask_w = false;  // num_inner % tt::constants::TILE_WIDTH;
    uint32_t block_size = get_block_size(Wt);

    // 2) Create and configure circular buffers
    auto cb_input = create_circular_buffer(program, all_cores, kInputCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_mask_w = create_circular_buffer(program, all_cores, kMaskWCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_scaler = create_circular_buffer(program, all_cores, kScalerCbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_gamma = create_circular_buffer(program, all_cores, kGammaCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_rms_a = create_circular_buffer(program, all_cores, kRmsACbIndex, data_format, single_tile_size_bytes, 2);
    auto cb_dLdout = create_circular_buffer(program, all_cores, kDLoutCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_dL_da = create_circular_buffer(program, all_cores, kDLdaCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_dL_dgamma =
        create_circular_buffer(program, all_cores, kDLdgammaCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_scaled_gain =
        create_circular_buffer(program, all_cores, kScaledGainCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_gained_dL_dout =
        create_circular_buffer(program, all_cores, kGainedDLdoutCbIndex, data_format, single_tile_size_bytes, Wt);
    auto cb_scale = create_circular_buffer(program, all_cores, kScaleCbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_ms_a = create_circular_buffer(program, all_cores, kMsACbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_c_by_ms_a =
        create_circular_buffer(program, all_cores, kCByMsACbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_rhs = create_circular_buffer(program, all_cores, kRhsCbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_a_over_rms_a =
        create_circular_buffer(program, all_cores, kAOverRmsACbIndex, data_format, single_tile_size_bytes, 1);
    auto cb_dL_dgamma_components =
        create_circular_buffer(program, all_cores, kDLdgammaComponentsCbIndex, data_format, single_tile_size_bytes, Wt);

    // 3) Create reader/writer/compute kernels
    auto* input_buffer = input.buffer();
    auto* gamma_buffer = gamma.buffer();
    auto* rms_buffer = rms.buffer();
    auto* dLdout_buffer = dLdout.buffer();
    auto* da_buffer = output[0].buffer();
    auto* dgamma_buffer = output[1].buffer();

    // TODO: Configue defines like masking and fits in L1 etc.
    std::map<std::string, std::string> defines;  // Add defines as needed

    // Setup defines for reduce.
    // Compute kernel does not compile without these defines.
    // LLK reduction uses define values as default template parameters.
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

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
    kernels.compute = create_compute_kernel(
        program, all_cores, {num_rows_per_core, block_size, mask_w, Wt}, defines, kComputeKernelPath);

    // 4) Assign runtime args for each core
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        gamma_buffer,
        rms_buffer,
        dLdout_buffer,
        da_buffer,
        dgamma_buffer,
        num_cores,
        num_cores_y,
        num_rows_per_core,
        all_cores);

    // 5) Return the fully configured program & relevant shared variables
    return cached_program_t{std::move(program), {/* Add any shared variables needed for your backward op here */}};
}

void RMSNormBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // For now, do nothing (dummy)
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device
