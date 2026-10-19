// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

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
constexpr uint32_t kExternalSumsBufferIdx = 6U;

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
constexpr auto kZeroCbIndex = tt::CBIndex::c_7;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_8;
constexpr auto kDLdgammaComponentsCbIndex = tt::CBIndex::c_9;
// CBs with intermediate computations
constexpr auto kRecipRmsACbIndex = tt::CBIndex::c_10;
constexpr auto kScaleCbIndex = tt::CBIndex::c_11;
constexpr auto kExternalSumsCbIndex = tt::CBIndex::c_12;
constexpr auto kMaskHCbIndex = tt::CBIndex::c_13;

// Some of the below constants are set to 2U because we might need to push a new value before popping the old one.
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumMaskHTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsATiles = 2U;
constexpr uint32_t kNumMatMulReduceTiles = 1U;
constexpr uint32_t kNumRecipRmsATiles = 1U;
constexpr uint32_t kNumScaleTiles = 2U;
constexpr uint32_t kNumZeroTiles = 1U;
constexpr uint32_t kNumLocalSumTiles = 2U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kLocalSumDefineKey = "LOCAL_SUM";
const std::string kExternalScaleDefineKey = "EXTERNAL_SCALE";
const std::string kDgammaOnChipDefineKey = "DGAMMA_ON_CHIP";
const std::string kMaskHDefineKey = "DO_MASK_H";

uint32_t num_external_sum_tiles(const ttml::metal::ops::rmsnorm_bw::device::tensor_args_t& tensor_args) {
    return tensor_args.external_sums.has_value()
               ? tensor_args.external_sums->padded_shape()[-1] / tt::constants::TILE_WIDTH
               : 0U;
}

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const RMSNormBackwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* gamma_buffer,
    const tt::tt_metal::Buffer* rms_buffer,
    const tt::tt_metal::Buffer* dLdout_buffer,
    const tt::tt_metal::Buffer* external_sums_buffer,
    const tt::tt_metal::Buffer* da_buffer,
    const tt::tt_metal::Buffer* dgamma_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many rows this core will process
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        // Reader kernel: (input_addr, gamma_addr, rms_addr, dLdout_addr, num_rows, offset, external_sums_addr)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(),
             gamma_buffer->address(),
             rms_buffer->address(),
             dLdout_buffer->address(),
             num_rows_per_core,
             num_rows_written,
             external_sums_buffer->address()});

        // Writer kernel: (da_addr, dgamma_addr, num_rows, offset, dgamma_tile_row)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {da_buffer->address(), dgamma_buffer->address(), num_rows_per_core, num_rows_written, i});

        // Compute kernel: (offset)
        SetRuntimeArgs(
            program,
            core_group_1.contains(core) ? kernels.compute_group_1 : kernels.compute_group_2,
            core,
            {num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

uint64_t required_l1_bytes(
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t num_external_tiles,
    const bool everything_fits_in_l1,
    const bool dgamma_on_chip) {
    const uint64_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint64_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t twice_block_size = 2U * block_size;
    const uint32_t num_row_tiles = everything_fits_in_l1 ? Wt : twice_block_size;

    // Memory of input tensors: input, gamma and dL_dout hold a whole row when it fits in L1.
    const uint64_t row_inputs_memory = 3U * num_row_tiles * bfloat16_single_tile_size_bytes;
    const uint64_t constant_tiles_memory =
        (kNumMaskTiles + kNumMaskHTiles + kNumScalerTiles + kNumMatMulReduceTiles + kNumZeroTiles) *
        bfloat16_single_tile_size_bytes;
    const uint64_t rms_a_memory = kNumRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t external_sums_memory = 2U * num_external_tiles * float32_single_tile_size_bytes;
    // dL_da is drained block-by-block by the writer, so it is double-buffered independent of Wt. dgamma is either
    // drained the same way or accumulated over the whole row in fp32.
    const uint64_t dL_da_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dgamma_memory =
        dgamma_on_chip ? Wt * float32_single_tile_size_bytes : twice_block_size * bfloat16_single_tile_size_bytes;
    // Memory for intermediate computations
    const uint64_t recip_rms_a_bcasted_memory = kNumRecipRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t scale_memory = kNumScaleTiles * float32_single_tile_size_bytes;

    return row_inputs_memory + constant_tiles_memory + rms_a_memory + external_sums_memory + dL_da_memory +
           dL_dgamma_memory + recip_rms_a_bcasted_memory + scale_memory;
}

RMSNormBackwardLayout compute_layout(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto* device = input.device();
    const auto& padded_shape = input.padded_shape();
    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t total_rows_to_process = padded_shape[0] * padded_shape[1] * Ht;
    const uint32_t block_size = get_block_size(Wt, 2U);  // We need two extra registers during calculation
    const uint32_t num_external_tiles = num_external_sum_tiles(tensor_args);

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::get<0>(tt::tt_metal::split_work_to_cores(grid, total_rows_to_process));

    const uint64_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto fits = [&](bool everything_fits_in_l1, bool dgamma_on_chip) {
        return required_l1_bytes(Wt, block_size, num_external_tiles, everything_fits_in_l1, dgamma_on_chip) <=
               available_L1_in_bytes;
    };

    // The on-chip dgamma accumulator replaces writing [B,N,S,C] components and reading them back in ttnn::sum,
    // which costs more than the second read of a streamed row, so it is preferred over keeping the row in L1.
    const bool can_accumulate_dgamma = args.mode != RMSNormBackwardMode::LocalSum;
    for (const auto [everything_fits_in_l1, dgamma_on_chip] :
         {std::pair{true, true}, std::pair{false, true}, std::pair{true, false}}) {
        if ((!dgamma_on_chip || can_accumulate_dgamma) && fits(everything_fits_in_l1, dgamma_on_chip)) {
            return RMSNormBackwardLayout{
                .num_cores = num_cores,
                .everything_fits_in_l1 = everything_fits_in_l1,
                .dgamma_on_chip = dgamma_on_chip,
            };
        }
    }
    return RMSNormBackwardLayout{.num_cores = num_cores, .everything_fits_in_l1 = false, .dgamma_on_chip = false};
}

RMSNormBackwardProgramFactory::cached_program_t RMSNormBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& dLdout = tensor_args.dL_dout;
    const bool local_sum = args.mode == RMSNormBackwardMode::LocalSum;
    const bool external_scale = args.mode == RMSNormBackwardMode::ExternalScale;

    // Check input shape is [B, N, S, C]
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);

    // Check gamma shape is [1, 1, 1, C]
    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(gamma_shape.rank() == 4, "Gamma tensor must be 4D [1, 1, 1, C], got shape {}", gamma_shape);
    TT_FATAL(
        gamma_shape[0] == 1 && gamma_shape[1] == 1 && gamma_shape[2] == 1,
        "Gamma tensor must have shape [1, 1, 1, C], got shape {}",
        gamma_shape);

    // Check C matches between input and gamma
    TT_FATAL(
        input_shape[3] == gamma_shape[3],
        "Gamma last dim (C) must match input last dim (C): input C={}, gamma C={}",
        input_shape[3],
        gamma_shape[3]);

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

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
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    uint32_t block_size = get_block_size(Wt, 2U);  // We need two extra registers during calculation

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // The dx formula divides the reduced scale by the width of the whole row, which is wider than the local shard in
    // ExternalScale mode. The scaler is negative because the compute kernel folds it into the per-row scale tile and
    // subtracts the a * scale term by accumulating it on the FPU (see compute_dL_da_block).
    const uint32_t row_width = external_scale ? args.full_width : num_inner;
    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(-1.F / row_width));
    const uint32_t num_external_tiles = num_external_sum_tiles(tensor_args);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const auto layout = compute_layout(args, tensor_args);
    const bool everything_fits_in_l1 = layout.everything_fits_in_l1;
    const bool dgamma_on_chip = layout.dgamma_on_chip;
    TT_FATAL(
        layout.num_cores == num_cores, "Layout core count {} differs from work split {}", layout.num_cores, num_cores);
    // Rows past the logical sequence length in the last tile row of each sequence are excluded from on-chip dgamma.
    const uint32_t mask_h = input.logical_shape()[-2] % tt::constants::TILE_HEIGHT;
    const bool do_mask_h = dgamma_on_chip && mask_h != 0U;

    const uint32_t num_input_tiles =
        (everything_fits_in_l1) ? Wt : twice_block_size;  // If everything fits in L1, read Wt tiles, else read 2x block

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;

    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    [[maybe_unused]] auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_rms_a = create_circular_buffer(
        program, all_cores, kRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsATiles);
    [[maybe_unused]] auto cb_dLdout = create_circular_buffer(
        program, all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMatMulReduceTiles);
    [[maybe_unused]] auto cb_zero = create_circular_buffer(
        program, all_cores, kZeroCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumZeroTiles);
    // Output CBs are produced and drained one block at a time (see writer: "interleave the writes to avoid waiting
    // for the entire Wt tiles at once"), so they only need double-buffering, not a full row. Sizing them to
    // num_input_tiles (= Wt on the fits-in-L1 path) overflows L1 for large hidden dims (e.g. 32B, Wt=160).
    // LocalSum writes one fp32 tile of row sums per row through the dL_da CB.
    [[maybe_unused]] auto cb_dL_da =
        local_sum
            ? create_circular_buffer(
                  program,
                  all_cores,
                  kDLdaCbIndex,
                  precise_data_format,
                  float32_single_tile_size_bytes,
                  kNumLocalSumTiles)
            : create_circular_buffer(
                  program, all_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_dL_dgamma_components = dgamma_on_chip ? create_circular_buffer(
                                                                         program,
                                                                         all_cores,
                                                                         kDLdgammaComponentsCbIndex,
                                                                         precise_data_format,
                                                                         float32_single_tile_size_bytes,
                                                                         Wt)
                                                                   : create_circular_buffer(
                                                                         program,
                                                                         all_cores,
                                                                         kDLdgammaComponentsCbIndex,
                                                                         data_format,
                                                                         bfloat16_single_tile_size_bytes,
                                                                         twice_block_size);
    if (do_mask_h) {
        create_circular_buffer(
            program, all_cores, kMaskHCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskHTiles);
    }
    [[maybe_unused]] auto cb_recip_rms_a_bcasted = create_circular_buffer(
        program, all_cores, kRecipRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRecipRmsATiles);
    [[maybe_unused]] auto cb_scale = create_circular_buffer(
        program, all_cores, kScaleCbIndex, precise_data_format, float32_single_tile_size_bytes, kNumScaleTiles);
    if (external_scale) {
        create_circular_buffer(
            program,
            all_cores,
            kExternalSumsCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            2U * num_external_tiles);
    }

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto check_dram = [](const tt::tt_metal::Buffer* buffer, const char* name) {
        TT_FATAL(
            buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "{} buffer must be in DRAM. {} buffer of type {}",
            name,
            name,
            enchantum::to_string(buffer->buffer_type()));
    };

    auto* input_buffer = input.buffer();
    check_dram(input_buffer, "Input");
    auto* gamma_buffer = gamma.buffer();
    check_dram(gamma_buffer, "Gamma");
    auto* dLdout_buffer = dLdout.buffer();
    check_dram(dLdout_buffer, "dL_dout");

    // Tensors a mode does not read still need a valid buffer for the compile-time accessor args, so they alias
    // the input; the kernels never touch them.
    auto* rms_buffer = tensor_args.rms.has_value() ? tensor_args.rms->buffer() : input_buffer;
    check_dram(rms_buffer, "RMS");
    auto* external_sums_buffer = external_scale ? tensor_args.external_sums->buffer() : input_buffer;
    check_dram(external_sums_buffer, "External sums");

    auto* dL_da_buffer = output[0].buffer();
    check_dram(dL_da_buffer, "dL_da");
    auto* dL_dgamma_components_buffer = local_sum ? dL_da_buffer : output[1].buffer();
    check_dram(dL_dgamma_components_buffer, "dL_dgamma");

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }
    if (local_sum) {
        defines[kLocalSumDefineKey] = "1";
    }
    if (external_scale) {
        defines[kExternalScaleDefineKey] = "1";
    }
    if (dgamma_on_chip) {
        defines[kDgammaOnChipDefineKey] = "1";
    }
    if (do_mask_h) {
        defines[kMaskHDefineKey] = "1";
    }
    RMSNormBackwardKernels kernels;
    std::vector<uint32_t> reader_compile_time_args{packed_scaler, block_size, mask_w, Wt, num_external_tiles, mask_h};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(rms_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dLdout_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(external_sums_buffer).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_da_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dL_dgamma_components_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for rmsnorm_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        mask_w,                     // mask_w
        Wt,                         // num_inner / TILE_W
        num_external_tiles,         // tiles of external sums per row
        Ht                          // tile rows per sequence
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            mask_w,                     // mask_w
            Wt,                         // num_inner / TILE_W
            num_external_tiles,         // tiles of external sums per row
            Ht                          // tile rows per sequence
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
        rms_buffer,
        dLdout_buffer,
        external_sums_buffer,
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
        {/* rmsnorm_bw_reader_kernel_id  = */ kernels.reader,
         /* rmsnorm_bw_writer_kernel_id  = */ kernels.writer,
         /* rmsnorm_bw_kernel_group_1_id = */ kernels.compute_group_1,
         /* rmsnorm_bw_kernel_group_2_id = */ kernels.compute_group_2,
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

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* gamma_buffer = tensor_args.gamma.buffer();
    auto* rms_buffer = tensor_args.rms.has_value() ? tensor_args.rms->buffer() : input_buffer;
    auto* dLdout_buffer = tensor_args.dL_dout.buffer();
    auto* external_sums_buffer =
        tensor_args.external_sums.has_value() ? tensor_args.external_sums->buffer() : input_buffer;

    auto* da_buffer = output[0].buffer();
    auto* dgamma_buffer = output.size() > 1U ? output[1].buffer() : da_buffer;

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kGammaBufferIdx] = gamma_buffer->address();
            runtime_args[kRmsBufferIdx] = rms_buffer->address();
            runtime_args[kDLdoutBufferIdx] = dLdout_buffer->address();
            runtime_args[kExternalSumsBufferIdx] = external_sums_buffer->address();
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
