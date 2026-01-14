// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_q_program_factory.hpp"

#include <bit>
#include <cmath>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_q_reader_kernel.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_q_writer_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_q_compute_kernel.cpp";

// Reader runtime args
constexpr uint32_t kGradOutputBufferIdx = 0;
constexpr uint32_t kAttnOutputBufferIdx = 1U;
constexpr uint32_t kQueryBufferIdx = 2U;
constexpr uint32_t kKeyBufferIdx = 3U;
constexpr uint32_t kValueBufferIdx = 4U;
constexpr uint32_t kAttnMaskBufferIdx = 5U;
constexpr uint32_t kIntermediatesBufferIdx = 6U;

// Writer runtime args
constexpr uint32_t kGradQueryBufferIdx = 0;

// Circular buffer indices
constexpr auto kGradOutputCbIndex = tt::CBIndex::c_0;
constexpr auto kAttnOutputCbIndex = tt::CBIndex::c_1;
constexpr auto kQueryCbIndex = tt::CBIndex::c_2;
constexpr auto kKeyCbIndex = tt::CBIndex::c_3;
constexpr auto kValueCbIndex = tt::CBIndex::c_4;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_5;
constexpr auto kIntermediatesCbIndex = tt::CBIndex::c_6;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_7;
constexpr auto kPrevGradQueryHolderCbIndex = tt::CBIndex::c_8;
constexpr auto kCurGradQueryHolderCbIndex = tt::CBIndex::c_9;
constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_10;
constexpr auto kGradAttentionCbIndex = tt::CBIndex::c_11;
constexpr auto kGradScoresCbIndex = tt::CBIndex::c_12;
constexpr auto kTransposeWhCbIndex = tt::CBIndex::c_13;
constexpr auto kUScalarRowCbIndex = tt::CBIndex::c_14;
constexpr auto kGradQueryCbIndex = tt::CBIndex::c_15;

constexpr uint32_t kSingleTileBuffer = 1U;
constexpr uint32_t kNumOfIntermCBTiles = 2U;

const std::string kFP32DestAccEnKey = "FP32_DEST_ACC_EN";

}  // namespace

namespace ttml::metal::ops::sdpa_bw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct SDPABackwardQKernels {
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
    const SDPABackwardQKernels& kernels,
    const tt::tt_metal::Buffer* grad_output_buffer,
    const tt::tt_metal::Buffer* attn_output_buffer,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* attn_mask_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    const tt::tt_metal::Buffer* grad_query_buffer,
    const uint32_t num_cores,
    const uint32_t num_cores_y,
    const uint32_t num_rows_per_core_group_1,
    const uint32_t num_rows_per_core_group_2,
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

        // Reader kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {
                grad_output_buffer->address(),                                   // grad_output buffer address
                attn_output_buffer->address(),                                   // attn_output buffer address
                query_buffer->address(),                                         // query buffer address
                key_buffer->address(),                                           // key buffer address
                value_buffer->address(),                                         // value buffer address
                attn_mask_buffer != nullptr ? attn_mask_buffer->address() : 0U,  // mask buffer address
                intermediates_buffer->address(),                                 // intermediates buffer address
                num_rows_per_core,                                               // rows to process in this kernel
                num_rows_written                                                 // starting row for this core
            });

        // Writer kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {
                grad_query_buffer->address(),  // grad_query buffer address
                num_rows_per_core,             // rows to process in this kernel
                num_rows_written               // starting row for this core
            });

        num_rows_written += num_rows_per_core;
    }
}

SDPABackwardQProgramFactory::cached_program_t SDPABackwardQProgramFactory::create(
    const q::operation_attributes_t& args, const q::tensor_args_t& tensor_args, q::tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& grad_output = tensor_args.grad_output;
    const auto& attn_output = tensor_args.attn_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;

    auto* device = grad_output.device();
    tt::tt_metal::Program program{};
    auto input_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // Get tensor dimensions and extract heads from shapes
    const auto [qB, qNH, qS, qEmbd] = grad_output.padded_shape().to_array_4D();
    const auto [kB, kNH, kS, kEmbd] = key.padded_shape().to_array_4D();
    const auto [vB, vNH, vS, vEmbd] = value.padded_shape().to_array_4D();

    // For query backward pass we split work over rows of Q
    const uint32_t St = qS / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    const uint32_t NC = qB * qNH;
    const uint32_t total_rows_to_process =
        NC * St;                   // total rows to process = batch_size * num_heads * num_tiles_in_seq_len
    const uint32_t q_heads = qNH;  // number of heads in Query
    const uint32_t kv_heads = kNH;
    const uint32_t heads_per_group =
        qNH / kv_heads;  // we read one group of K and V for every heads_per_group heads from Q

    const uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    const uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    const uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;

    // Scale factor for attention computation
    // Note: qEmbd is already the per-head dimension (tensor shape is B, NH, S, Embd)
    const float per_head_dim = static_cast<float>(qEmbd);
    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(per_head_dim));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_grad_output = create_circular_buffer( // CBIndex::c_0
        program, all_cores, kGradOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_attn_output = create_circular_buffer( // CBIndex::c_1
        program, all_cores, kAttnOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_query = create_circular_buffer( // CBIndex::c_2
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_key =  // CBIndex::c_3
        create_circular_buffer(program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt);

    [[maybe_unused]] auto cb_value =  // CBIndex::c_4
        create_circular_buffer(
            program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

    [[maybe_unused]] auto cb_attn_mask =  // CBIndex::c_5
        create_circular_buffer(
            program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kSingleTileBuffer);

    [[maybe_unused]] auto cb_intermediates =  // CBIndex::c_6
        create_circular_buffer(
            program,
            all_cores,
            kIntermediatesCbIndex,
            data_format,
            bfloat16_single_tile_size_bytes,
            2 * kNumOfIntermCBTiles);

    [[maybe_unused]] auto cb_mat_mul_reduce =  // CBIndex::c_7
        create_circular_buffer(
            program,
            all_cores,
            kMatMulReduceCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_prev_grad_query =  // CBIndex::c_9
        create_circular_buffer(
            program, all_cores, kPrevGradQueryHolderCbIndex, precise_data_format, float32_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_cur_grad_query =  // CBIndex::c_10
        create_circular_buffer(
            program, all_cores, kCurGradQueryHolderCbIndex, precise_data_format, float32_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_attention_weights =  // CBIndex::c_13
        create_circular_buffer(
            program,
            all_cores,
            kAttentionWeightsCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_attention =  // CBIndex::c_14
        create_circular_buffer(
            program,
            all_cores,
            kGradAttentionCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_scores =  // CBIndex::c_15
        create_circular_buffer(
            program,
            all_cores,
            kGradScoresCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_transpose_wh =  // CBIndex::c_16
        create_circular_buffer(
            program, all_cores, kTransposeWhCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_u_scaler_row =  // CBIndex::c_17
        create_circular_buffer(
            program,
            all_cores,
            kUScalarRowCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_query =  // CBIndex::c_18
        create_circular_buffer(
            program, all_cores, kGradQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* grad_output_buffer = grad_output.buffer();
    auto* attn_output_buffer = attn_output.buffer();
    auto* query_buffer = query.buffer();
    auto* key_buffer = key.buffer();
    auto* value_buffer = value.buffer();
    auto* attn_mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask.value().buffer() : nullptr;
    auto* intermediates_buffer = intermediates.buffer();

    auto* grad_query_buffer = output.buffer();  // [grad_Q]

    // Configure defines
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    if (args.fp32_dest_acc_en) {
        defines[kFP32DestAccEnKey] = "1";
    }

    SDPABackwardQKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // 0: query width in tiles (also used for K/V since qWt == kWt == vWt)
        St,               // 1: sequence length in tiles
        q_heads,          // 2: number of query heads
        heads_per_group,  // 3: heads per group
    };
    tt::tt_metal::TensorAccessorArgs(grad_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(attn_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(attn_mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, defines, kReaderKernelPath);

    // Writer compile-time arguments
    std::vector<uint32_t> writer_compile_args = {
        qWt,  // 0: query width in tiles
    };
    tt::tt_metal::TensorAccessorArgs(grad_query_buffer).append_to(writer_compile_args);

    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // Set UnpackToDestFp32 only for accumulator buffers (used with SFPU/copy, not FPU matmul)
    auto create_unpack_to_dest_mode = []() {
        std::vector<UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        mode[tt::CBIndex::c_8] = UnpackToDestMode::UnpackToDestFp32;  // kPrevGradQueryHolderCbIndex
        mode[tt::CBIndex::c_9] = UnpackToDestMode::UnpackToDestFp32;  // kCurGradQueryHolderCbIndex
        return mode;
    };

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // 0: per_core_block_cnt
        qWt,                        // 1: num tile in inner dim (qWt == kWt == vWt)
        St,                         // 2: num_seq_len / TILE_H
        scaler,                     // 3: sqrt(Et) - sdpa scaler factor
        minus_one,                  // 4: used to transform mask from 1/0 to 0/-1
        custom_inf                  // 5: used to transform mask from 0/-1 to 0/-inf
    };
    kernels.compute_group_1 = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = create_unpack_to_dest_mode(),
            .math_approx_mode = false,
            .compile_args = compute_group_1_args,
            .defines = defines});

    // Group 2 compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // 0: per_core_block_cnt
            qWt,                        // 1: num tile in inner dim (qWt == kWt == vWt)
            St,                         // 2: num_seq_len / TILE_H
            scaler,                     // 3: sqrt(Et) - sdpa scaler factor
            minus_one,                  // 4: used to transform mask from 1/0 to 0/-1
            custom_inf                  // 5: used to transform mask from 0/-1 to 0/-inf
        };
        kernels.compute_group_2 = tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = create_unpack_to_dest_mode(),
                .math_approx_mode = false,
                .compile_args = compute_group_2_args,
                .defines = defines});
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        grad_output_buffer,
        attn_output_buffer,
        query_buffer,
        key_buffer,
        value_buffer,
        attn_mask_buffer,
        intermediates_buffer,
        grad_query_buffer,
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
        {.sdpa_bw_q_reader_kernel = kernels.reader,
         .sdpa_bw_q_writer_kernel = kernels.writer,
         .sdpa_bw_q_kernel_group_1 = kernels.compute_group_1,
         .sdpa_bw_q_kernel_group_2 = kernels.compute_group_2,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void SDPABackwardQProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const q::operation_attributes_t& args,
    const q::tensor_args_t& tensor_args,
    q::tensor_return_value_t& tensor_return_value) {
    // This updates buffer addresses and other runtime parameters
    auto& shared_vars = cached_program.shared_variables;
    auto& sdpa_bw_reader_kernel = shared_vars.sdpa_bw_q_reader_kernel;
    auto& sdpa_bw_writer_kernel = shared_vars.sdpa_bw_q_writer_kernel;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto* grad_output_buffer = tensor_args.grad_output.buffer();
    const auto* attn_output_buffer = tensor_args.attn_output.buffer();
    const auto* query_buffer = tensor_args.query.buffer();
    const auto* key_buffer = tensor_args.key.buffer();
    const auto* value_buffer = tensor_args.value.buffer();
    const auto* mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask.value().buffer() : nullptr;
    const auto* intermediates_buffer = tensor_args.intermediates.buffer();

    const auto* grad_query_buffer = tensor_return_value.buffer();  // [grad_Q]

    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_bw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_bw_writer_kernel);

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kGradOutputBufferIdx] = grad_output_buffer->address();
            runtime_args[kAttnOutputBufferIdx] = attn_output_buffer->address();
            runtime_args[kQueryBufferIdx] = query_buffer->address();
            runtime_args[kKeyBufferIdx] = key_buffer->address();
            runtime_args[kValueBufferIdx] = value_buffer->address();
            runtime_args[kAttnMaskBufferIdx] = mask_buffer != nullptr ? mask_buffer->address() : 0U;
            runtime_args[kIntermediatesBufferIdx] = intermediates_buffer->address();
        }

        // Update output buffer for writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kGradQueryBufferIdx] = grad_query_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::sdpa_bw::device
