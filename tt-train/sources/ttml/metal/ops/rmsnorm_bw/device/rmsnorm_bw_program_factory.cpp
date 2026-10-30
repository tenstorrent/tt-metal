// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/ops/common/rmsnorm_common.hpp"

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
constexpr uint32_t kGateBufferIdx = 4U;

// Writer buffer indices
constexpr uint32_t kDaBufferIdx = 0;
constexpr uint32_t kDgammaComponentsBufferIdx = 1U;
constexpr uint32_t kDgateBufferIdx = 2U;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kMaskWCbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kGammaCbIndex = tt::CBIndex::c_3;
constexpr auto kRmsACbIndex = tt::CBIndex::c_4;
constexpr auto kDLoutCbIndex = tt::CBIndex::c_5;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;
constexpr auto kZeroCbIndex = tt::CBIndex::c_7;
constexpr auto kGateCbIndex = tt::CBIndex::c_12;
// CBs with output data
constexpr auto kDLdaCbIndex = tt::CBIndex::c_8;
constexpr auto kDLdgammaComponentsCbIndex = tt::CBIndex::c_9;
constexpr auto kDLdgateCbIndex = tt::CBIndex::c_13;
// CBs with intermediate computations
constexpr auto kRecipRmsACbIndex = tt::CBIndex::c_10;
constexpr auto kScaleCbIndex = tt::CBIndex::c_11;
constexpr auto kDuCbIndex = tt::CBIndex::c_14;
constexpr auto kSlicePartialCbIndex = tt::CBIndex::c_15;
constexpr auto kGatheredPartialsCbIndex = tt::CBIndex::c_16;

// Some of the below constants are set to 2U because we might need to push a new value before popping the old one.
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumRmsATiles = 2U;
constexpr uint32_t kNumMatMulReduceTiles = 1U;
constexpr uint32_t kNumRecipRmsATiles = 1U;
constexpr uint32_t kNumScaleTiles = 2U;
constexpr uint32_t kNumZeroTiles = 1U;
constexpr uint32_t kNumSlicePartialTiles = 1U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";
const std::string kComputeDgammaDefineKey = "COMPUTE_DGAMMA";
const std::string kGatedDefineKey = "GATED";

}  // namespace

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

struct RMSNormBackwardBuffers {
    const tt::tt_metal::Buffer* input;
    const tt::tt_metal::Buffer* gamma;
    const tt::tt_metal::Buffer* rms;
    const tt::tt_metal::Buffer* dL_dout;
    const tt::tt_metal::Buffer* gate;  // nullptr unless gated
    const tt::tt_metal::Buffer* da;
    const tt::tt_metal::Buffer* dgamma_components;  // nullptr unless compute_dgamma
    const tt::tt_metal::Buffer* dgate;              // nullptr unless gated
};

RMSNormBackwardBuffers get_buffers(const tensor_args_t& tensor_args, const tensor_return_value_t& output) {
    return RMSNormBackwardBuffers{
        .input = tensor_args.input.buffer(),
        .gamma = tensor_args.gamma.buffer(),
        .rms = tensor_args.rms.buffer(),
        .dL_dout = tensor_args.dL_dout.buffer(),
        .gate = tensor_args.gate.has_value() ? tensor_args.gate->buffer() : nullptr,
        .da = output[0]->buffer(),
        .dgamma_components = output[1].has_value() ? output[1]->buffer() : nullptr,
        .dgate = output[2].has_value() ? output[2]->buffer() : nullptr,
    };
}

// Wt is the tiles of each row a core holds, its slice with num_slices > 1, which adds the partial exchange.
bool fits_in_l1_check(
    const uint32_t Wt,
    const uint32_t num_slices,
    const uint32_t input_single_tile_size_bytes,
    const uint32_t bfloat16_single_tile_size_bytes,
    const uint32_t float32_single_tile_size_bytes,
    const bool compute_dgamma,
    const bool gated,
    const uint64_t available_L1_in_bytes) {
    const uint32_t twice_block_size = 2U * get_block_size(Wt, 2U);

    // Memory of input tensors
    const uint64_t input_memory = Wt * input_single_tile_size_bytes;
    const uint64_t mask_memory = kNumMaskTiles * bfloat16_single_tile_size_bytes;
    const uint64_t scaler_memory = kNumScalerTiles * bfloat16_single_tile_size_bytes;
    const uint64_t gamma_memory = Wt * bfloat16_single_tile_size_bytes;
    const uint64_t rms_a_memory = kNumRmsATiles * bfloat16_single_tile_size_bytes;
    // Gated mode consumes dL_dout and the gate block by block, turning them into du, which holds the whole row.
    const uint64_t dL_dout_memory = (gated ? twice_block_size : Wt) * bfloat16_single_tile_size_bytes;
    const uint64_t gate_memory = gated ? twice_block_size * bfloat16_single_tile_size_bytes : 0U;
    const uint64_t du_memory = gated ? Wt * float32_single_tile_size_bytes : 0U;
    const uint64_t matmul_reduce_memory = kNumMatMulReduceTiles * bfloat16_single_tile_size_bytes;
    const uint64_t zero_memory = kNumZeroTiles * bfloat16_single_tile_size_bytes;
    // Memory for output tensors. The output CBs are drained block-by-block by the writer, so they are
    // double-buffered (twice_block_size), independent of Wt.
    const uint64_t dL_da_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t dL_dgamma_components_memory =
        compute_dgamma ? twice_block_size * bfloat16_single_tile_size_bytes : 0U;
    const uint64_t dL_dgate_memory = gated ? twice_block_size * bfloat16_single_tile_size_bytes : 0U;
    // Memory for intermediate computations
    const uint64_t recip_rms_a_bcasted_memory = kNumRecipRmsATiles * bfloat16_single_tile_size_bytes;
    const uint64_t scale_memory = kNumScaleTiles * float32_single_tile_size_bytes;
    const uint64_t partials_memory =
        num_slices > 1U ? (kNumSlicePartialTiles + 2U * num_slices) * float32_single_tile_size_bytes : 0U;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes = input_memory + mask_memory + scaler_memory + gamma_memory + rms_a_memory +
                                          dL_dout_memory + gate_memory + du_memory + matmul_reduce_memory +
                                          dL_da_memory + dL_dgamma_components_memory + dL_dgate_memory +
                                          recip_rms_a_bcasted_memory + scale_memory + zero_memory + partials_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

RMSNormBackwardProgramFactory::cached_program_t RMSNormBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& gate = tensor_args.gate;
    const bool gated = gate.has_value();
    const bool compute_dgamma = args.compute_dgamma;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(
        input_data_format == tt::DataFormat::Float16_b || input_data_format == tt::DataFormat::Float32,
        "Input data format must be Float16_b or Float32");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);
    uint32_t input_single_tile_size_bytes = tt::tile_size(input_data_format);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // Gated mode reads dL_dout and the gate and writes dL_dgate in the head-merged layout; <1, 1> is the identity.
    const uint32_t layout_Ht = gated ? Ht : 1U;
    const uint32_t layout_num_heads = gated ? get_num_heads_of_head_merged(input, gate.value()) : 1U;

    // Get the number of inner dimension
    uint32_t num_inner = input.logical_shape()[-1];

    // This parameter is used to determine if we need to mask tiles, i.e. if the operation applied over inner dimension
    // might produce incorrect results due to some random data in the end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    const uint64_t available_L1_in_bytes = available_l1_bytes(*device);
    const auto fits_in_l1 = [&](const uint32_t row_tiles, const uint32_t num_slices) {
        return fits_in_l1_check(
            row_tiles,
            num_slices,
            input_single_tile_size_bytes,
            bfloat16_single_tile_size_bytes,
            float32_single_tile_size_bytes,
            compute_dgamma,
            gated,
            available_L1_in_bytes);
    };
    // Gated rows are single heads, already one work item per head.
    const auto plan = rmsnorm::plan_row_slices(
        compute_with_storage_grid_size,
        total_rows_to_process,
        Wt,
        /* max_slices */ gated ? 1U : args.max_row_slices,
        fits_in_l1);
    const uint32_t St = plan.slice_tiles;
    const uint32_t num_slices = plan.num_slices;
    const bool sliced = num_slices > 1U;
    const auto& all_cores = plan.all_cores;
    const auto& core_group_1 = plan.core_group_1;
    const auto& core_group_2 = plan.core_group_2;

    // Compile arguments
    uint32_t block_size = get_block_size(St, 2U);  // We need two extra registers during calculation

    uint32_t packed_scaler = pack_two_bfloat16_to_uint32(static_cast<float>(1.F / num_inner));

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    const bool everything_fits_in_l1 = fits_in_l1(St, num_slices);
    check_gated_row_fits_in_l1(gated, everything_fits_in_l1, Wt);

    const uint32_t num_input_tiles =
        (everything_fits_in_l1) ? St : twice_block_size;  // If everything fits in L1, read St tiles, else read 2x block
    // Gated mode consumes dL_dout block by block and keeps du = dL_dout * silu(gate) for the whole row instead.
    const uint32_t num_dL_dout_tiles = gated ? twice_block_size : num_input_tiles;

    auto data_format = tt::DataFormat::Float16_b;
    auto precise_data_format = tt::DataFormat::Float32;

    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, input_data_format, input_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);
    [[maybe_unused]] auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    [[maybe_unused]] auto cb_gamma = create_circular_buffer(
        program, all_cores, kGammaCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);
    [[maybe_unused]] auto cb_rms_a = create_circular_buffer(
        program, all_cores, kRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRmsATiles);
    [[maybe_unused]] auto cb_dLdout = create_circular_buffer(
        program, all_cores, kDLoutCbIndex, data_format, bfloat16_single_tile_size_bytes, num_dL_dout_tiles);
    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMatMulReduceTiles);
    [[maybe_unused]] auto cb_zero = create_circular_buffer(
        program, all_cores, kZeroCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumZeroTiles);
    // Output CBs are produced and drained one block at a time (see writer: "interleave the writes to avoid waiting
    // for the entire Wt tiles at once"), so they only need double-buffering, not a full row. Sizing them to
    // num_input_tiles (= Wt on the fits-in-L1 path) overflows L1 for large hidden dims (e.g. 32B, Wt=160).
    [[maybe_unused]] auto cb_dL_da = create_circular_buffer(
        program, all_cores, kDLdaCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    if (compute_dgamma) {
        [[maybe_unused]] auto cb_dL_dgamma_components = create_circular_buffer(
            program,
            all_cores,
            kDLdgammaComponentsCbIndex,
            data_format,
            bfloat16_single_tile_size_bytes,
            twice_block_size);
    }
    [[maybe_unused]] auto cb_recip_rms_a_bcasted = create_circular_buffer(
        program, all_cores, kRecipRmsACbIndex, data_format, bfloat16_single_tile_size_bytes, kNumRecipRmsATiles);
    [[maybe_unused]] auto cb_scale = create_circular_buffer(
        program, all_cores, kScaleCbIndex, precise_data_format, float32_single_tile_size_bytes, kNumScaleTiles);
    if (gated) {
        // The gate is read alongside dL_dout, so it is sized like it.
        [[maybe_unused]] auto cb_gate = create_circular_buffer(
            program, all_cores, kGateCbIndex, data_format, bfloat16_single_tile_size_bytes, num_dL_dout_tiles);
        [[maybe_unused]] auto cb_dL_dgate = create_circular_buffer(
            program, all_cores, kDLdgateCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
        [[maybe_unused]] auto cb_du = create_circular_buffer(
            program, all_cores, kDuCbIndex, precise_data_format, float32_single_tile_size_bytes, Wt);
    }
    uint32_t partials_sem_id = 0U;
    if (sliced) {
        [[maybe_unused]] auto cb_slice_partial = create_circular_buffer(
            program,
            all_cores,
            kSlicePartialCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kNumSlicePartialTiles);
        // Two rows of partials: a peer can send the next row's partial while this core still sums the current row.
        [[maybe_unused]] auto cb_gathered_partials = create_circular_buffer(
            program,
            all_cores,
            kGatheredPartialsCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            2U * num_slices);
        partials_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);
    }

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    const auto buffers = get_buffers(tensor_args, output);

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }
    if (compute_dgamma) {
        defines[kComputeDgammaDefineKey] = "1";
    }
    if (gated) {
        defines[kGatedDefineKey] = "1";
    }

    RMSNormBackwardKernels kernels;
    // An absent gate or output still appends a null accessor (two zero-valued args), so the offsets do not depend on
    // the mode.
    std::vector<uint32_t> reader_compile_time_args{
        packed_scaler, block_size, mask_w, Wt, layout_Ht, layout_num_heads, St};
    tt::tt_metal::TensorAccessorArgs(buffers.input).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.gamma).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.rms).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.dL_dout).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.gate).append_to(reader_compile_time_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{
        block_size, Wt, layout_Ht, layout_num_heads, St, num_slices, partials_sem_id};
    tt::tt_metal::TensorAccessorArgs(buffers.da).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.dgamma_components).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(buffers.dgate).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for rmsnorm_bw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        plan.rows_per_group_1,  // per_core_block_cnt
        block_size,             // per_core_block_size
        mask_w,                 // mask_w
        St,                     // num_inner / TILE_W
        num_slices};

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            plan.rows_per_group_2,  // per_core_block_cnt
            block_size,             // per_core_block_size
            mask_w,                 // mask_w
            St,                     // num_inner / TILE_W
            num_slices};

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }
    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    rmsnorm::for_each_row_slice_core(plan, [&](const rmsnorm::RowSliceWork& work) {
        // Reader kernel: (input_addr, gamma_addr, rms_addr, dLdout_addr, gate_addr, num_rows, offset, slice_start)
        SetRuntimeArgs(
            program,
            kernels.reader,
            work.core,
            {buffers.input->address(),
             buffers.gamma->address(),
             buffers.rms->address(),
             buffers.dL_dout->address(),
             get_address_or_zero(buffers.gate),
             work.num_rows,
             work.start_row,
             work.slice * St});
        // Writer kernel: (da_addr, dgamma_addr, dgate_addr, num_rows, offset, [slice, peer NOC x/y...])
        std::vector<uint32_t> writer_args{
            buffers.da->address(),
            get_address_or_zero(buffers.dgamma_components),
            get_address_or_zero(buffers.dgate),
            work.num_rows,
            work.start_row};
        const auto exchange_args = rmsnorm::row_slice_exchange_args(plan, work, *device);
        writer_args.insert(writer_args.end(), exchange_args.begin(), exchange_args.end());
        SetRuntimeArgs(program, kernels.writer, work.core, writer_args);
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
         /* num_cores                 = */ plan.num_cores,
         /* num_cores_y               = */ plan.num_cores_y}};
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

    const auto buffers = get_buffers(tensor_args, output);

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, rmsnorm_bw_writer_kernel_id);

    for_each_core(num_cores, num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = buffers.input->address();
            runtime_args[kGammaBufferIdx] = buffers.gamma->address();
            runtime_args[kRmsBufferIdx] = buffers.rms->address();
            runtime_args[kDLdoutBufferIdx] = buffers.dL_dout->address();
            runtime_args[kGateBufferIdx] = get_address_or_zero(buffers.gate);
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kDaBufferIdx] = buffers.da->address();
            runtime_args[kDgammaComponentsBufferIdx] = get_address_or_zero(buffers.dgamma_components);
            runtime_args[kDgateBufferIdx] = get_address_or_zero(buffers.dgate);
        }
    });
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device
