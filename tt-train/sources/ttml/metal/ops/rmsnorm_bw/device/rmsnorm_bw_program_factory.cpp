// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

#include "metal/common/program_utils.hpp"

namespace {

using ttml::metal::ops::rmsnorm_bw::device::operation_attributes_t;
using ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardMode;
using ttml::metal::ops::rmsnorm_bw::device::tensor_args_t;

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
constexpr auto kDgammaRowCbIndex = tt::CBIndex::c_14;

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
constexpr uint32_t kNumDgammaRowTiles = 2U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kLocalSumDefineKey = "LOCAL_SUM";
const std::string kExternalScaleDefineKey = "EXTERNAL_SCALE";
const std::string kDgammaOnChipDefineKey = "DGAMMA_ON_CHIP";
const std::string kMaskHDefineKey = "DO_MASK_H";

// Tensor-derived sizes shared by the layout choice and program creation.
struct RowGeometry {
    uint32_t Wt{};
    uint32_t Ht{};
    uint32_t num_rows{};  // tile rows over all batches and sequences
    uint32_t block_size{};
    uint32_t mask_w{};  // logical width % 32, 0 when the last tile is full
    uint32_t mask_h{};  // logical sequence length % 32, 0 when the last tile row is full
    uint32_t num_external_tiles{};
    tt::DataFormat data_format{};
};

RowGeometry row_geometry(const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& padded_shape = input.padded_shape();
    const auto& logical_shape = input.logical_shape();
    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    return RowGeometry{
        .Wt = Wt,
        .Ht = Ht,
        .num_rows = padded_shape[0] * padded_shape[1] * Ht,
        .block_size = get_block_size(Wt, 2U),  // We need two extra registers during calculation
        .mask_w = logical_shape[-1] % tt::constants::TILE_WIDTH,
        .mask_h = logical_shape[-2] % tt::constants::TILE_HEIGHT,
        .num_external_tiles = tensor_args.external_sums.has_value()
                                  ? tensor_args.external_sums->padded_shape()[-1] / tt::constants::TILE_WIDTH
                                  : 0U,
        .data_format = datatype_to_dataformat_converter(input.dtype()),
    };
}

struct CbAllocation {
    uint32_t index{};
    tt::DataFormat format{};
    uint32_t num_tiles{};
};

// Every CB the kernels use for this mode and layout. Both the L1 fit check in compute_layout and the CB creation in
// create() use this list.
std::vector<CbAllocation> cb_allocations(
    const RMSNormBackwardMode mode,
    const RowGeometry& geometry,
    const bool everything_fits_in_l1,
    const bool dgamma_on_chip) {
    const auto data_format = geometry.data_format;
    const auto precise_data_format = tt::DataFormat::Float32;
    // Output CBs are produced and drained one block at a time, so they only need double-buffering. Row inputs hold
    // the whole row when it fits in L1.
    const uint32_t twice_block_size = 2U * geometry.block_size;
    const uint32_t num_row_tiles = everything_fits_in_l1 ? geometry.Wt : twice_block_size;

    std::vector<CbAllocation> allocations{
        {kInputCbIndex, data_format, num_row_tiles},
        {kGammaCbIndex, data_format, num_row_tiles},
        {kDLoutCbIndex, data_format, num_row_tiles},
        {kScalerCbIndex, data_format, kNumScalerTiles},
        {kMatMulReduceCbIndex, data_format, kNumMatMulReduceTiles},
        {kZeroCbIndex, data_format, kNumZeroTiles},
        {kScaleCbIndex, precise_data_format, kNumScaleTiles},
    };
    if (geometry.mask_w != 0U) {
        allocations.push_back({kMaskWCbIndex, data_format, kNumMaskTiles});
    }
    if (mode == RMSNormBackwardMode::LocalSum) {
        // One fp32 tile of row sums per row, written through the dL_da CB.
        allocations.push_back({kDLdaCbIndex, precise_data_format, kNumLocalSumTiles});
        return allocations;
    }

    allocations.push_back({kRmsACbIndex, data_format, kNumRmsATiles});
    allocations.push_back({kRecipRmsACbIndex, data_format, kNumRecipRmsATiles});
    allocations.push_back({kDLdaCbIndex, data_format, twice_block_size});
    if (dgamma_on_chip) {
        // fp32 accumulator over all rows of a core, then one column-reduced row per tile for the writer.
        allocations.push_back({kDLdgammaComponentsCbIndex, precise_data_format, geometry.Wt});
        allocations.push_back({kDgammaRowCbIndex, precise_data_format, kNumDgammaRowTiles});
        if (geometry.mask_h != 0U) {
            allocations.push_back({kMaskHCbIndex, data_format, kNumMaskHTiles});
        }
    } else {
        allocations.push_back({kDLdgammaComponentsCbIndex, data_format, twice_block_size});
    }
    if (mode == RMSNormBackwardMode::ExternalScale) {
        allocations.push_back({kExternalSumsCbIndex, precise_data_format, 2U * geometry.num_external_tiles});
    }
    return allocations;
}

uint64_t required_l1_bytes(const std::vector<CbAllocation>& allocations) {
    uint64_t bytes = 0U;
    for (const auto& cb : allocations) {
        bytes += static_cast<uint64_t>(cb.num_tiles) * tt::tile_size(cb.format);
    }
    return bytes;
}

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

RMSNormBackwardLayout compute_layout(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto* device = tensor_args.input.device();
    const auto geometry = row_geometry(tensor_args);
    const uint32_t num_cores =
        std::get<0>(tt::tt_metal::split_work_to_cores(device->compute_with_storage_grid_size(), geometry.num_rows));

    const uint64_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // The on-chip dgamma accumulator is preferred over keeping the row in L1: it drops the [B,N,S,C] bf16
    // components output, which halves the DRAM peak of the backward. Where both layouts fit (C near 5120 on
    // Blackhole), the streamed row's second read makes it 10-20% slower than (fits, components).
    // A caller-provided components buffer is the output of this launch, so that path stays on per-row components
    // even when the accumulator would fit.
    const bool can_accumulate_dgamma =
        args.mode != RMSNormBackwardMode::LocalSum && !tensor_args.preallocated_dgamma_components.has_value();
    for (const auto [everything_fits_in_l1, dgamma_on_chip] :
         {std::pair{true, true}, std::pair{false, true}, std::pair{true, false}}) {
        if (dgamma_on_chip && !can_accumulate_dgamma) {
            continue;
        }
        const auto allocations = cb_allocations(args.mode, geometry, everything_fits_in_l1, dgamma_on_chip);
        if (required_l1_bytes(allocations) <= available_L1_in_bytes) {
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

    TT_FATAL(
        input.physical_volume() % tt::constants::TILE_HW == 0,
        "Padded input tensor volume must be divisible by TILE_HW");
    const auto geometry = row_geometry(tensor_args);
    const uint32_t Wt = geometry.Wt;
    const uint32_t block_size = geometry.block_size;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, geometry.num_rows);

    // The dx formula divides the reduced scale by the width of the whole row, which is wider than the local shard in
    // ExternalScale mode. The scaler is negative because the compute kernel folds it into the per-row scale tile and
    // subtracts the a * scale term by accumulating it on the FPU (see compute_dL_da_block).
    const uint32_t row_width = external_scale ? args.full_width : input_shape[-1];
    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(-1.F / row_width));

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const auto& layout = args.layout;
    const bool everything_fits_in_l1 = layout.everything_fits_in_l1;
    const bool dgamma_on_chip = layout.dgamma_on_chip;
    // Rows past the logical sequence length in the last tile row of each sequence are excluded from on-chip dgamma.
    const bool do_mask_h = dgamma_on_chip && geometry.mask_h != 0U;

    for (const auto& cb : cb_allocations(args.mode, geometry, everything_fits_in_l1, dgamma_on_chip)) {
        create_circular_buffer(program, all_cores, cb.index, cb.format, tt::tile_size(cb.format), cb.num_tiles);
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
    if (geometry.mask_w != 0) {
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
    std::vector<uint32_t> reader_compile_time_args{
        packed_scaler, block_size, geometry.mask_w, Wt, geometry.num_external_tiles, geometry.mask_h};
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
    // The dgamma accumulator is column-reduced on the SFPU, so it is unpacked straight to fp32 DEST.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (dgamma_on_chip) {
        unpack_to_dest_mode[kDLdgammaComponentsCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    auto create_compute = [&](const tt::tt_metal::CoreRangeSet& cores, const uint32_t num_rows_per_core) {
        const std::vector<uint32_t> compile_time_args = {
            num_rows_per_core,            // per_core_block_cnt
            block_size,                   // per_core_block_size
            geometry.mask_w,              // mask_w
            Wt,                           // num_inner / TILE_W
            geometry.num_external_tiles,  // tiles of external sums per row
            geometry.Ht                   // tile rows per sequence
        };
        return tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .math_approx_mode = false,
                .compile_args = compile_time_args,
                .defines = defines});
    };

    kernels.compute_group_1 = create_compute(core_group_1, num_rows_per_core_group_1);
    if (!core_group_2.ranges().empty()) {
        kernels.compute_group_2 = create_compute(core_group_2, num_rows_per_core_group_2);
    }
    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    for_each_core_with_work(
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        [&](const CoreWork& work) {
            const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
            // Reader kernel: (input_addr, gamma_addr, rms_addr, dLdout_addr, num_rows, offset, external_sums_addr)
            SetRuntimeArgs(
                program,
                kernels.reader,
                core,
                {input_buffer->address(),
                 gamma_buffer->address(),
                 rms_buffer->address(),
                 dLdout_buffer->address(),
                 num_rows,
                 start_row,
                 external_sums_buffer->address()});
            // Writer kernel: (da_addr, dgamma_addr, num_rows, offset, dgamma_row)
            SetRuntimeArgs(
                program,
                kernels.writer,
                core,
                {dL_da_buffer->address(), dL_dgamma_components_buffer->address(), num_rows, start_row, core_index});
            // Compute kernel: (offset)
            SetRuntimeArgs(
                program, in_group_1 ? kernels.compute_group_1 : kernels.compute_group_2, core, {start_row});
        });

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

    for_each_core(num_cores, num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
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
    });
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device
