// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_reader_kernel.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_writer_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_kernel.cpp";

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
constexpr uint32_t kGradKeyBufferIdx = 1U;
constexpr uint32_t kGradValueBufferIdx = 2U;

// Circular buffer indices
constexpr auto kGradOutputCbIndex = tt::CBIndex::c_0;
constexpr auto kAttnOutputCbIndex = tt::CBIndex::c_1;
constexpr auto kQueryCbIndex = tt::CBIndex::c_2;
constexpr auto kKeyCbIndex = tt::CBIndex::c_3;
constexpr auto kValueCbIndex = tt::CBIndex::c_4;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_5;
constexpr auto kIntermediatesCbIndex = tt::CBIndex::c_6;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_7;
constexpr auto kReductionScalerCbIndex = tt::CBIndex::c_8;

constexpr auto kPrevGradValueHolderCbIndex = tt::CBIndex::c_9;
constexpr auto kCurGradValueHolderCbIndex = tt::CBIndex::c_10;
constexpr auto kPrevGradKeyHolderCbIndex = tt::CBIndex::c_11;
constexpr auto kCurGradKeyHolderCbIndex = tt::CBIndex::c_12;

constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_13;
constexpr auto kGradAttentionCbIndex = tt::CBIndex::c_14;
constexpr auto kGradScoresCbIndex = tt::CBIndex::c_15;
constexpr auto kTransposeWhCbIndex = tt::CBIndex::c_16;
constexpr auto kUScalarRowCbIndex = tt::CBIndex::c_17;

constexpr auto kGradQueryCbIndex = tt::CBIndex::c_18;
constexpr auto kGradKeyCbIndex = tt::CBIndex::c_19;
constexpr auto kGradValueCbIndex = tt::CBIndex::c_20;
constexpr auto kSyncOutputWriterCbIndex = tt::CBIndex::c_21;

// [DEBUG]: Used for debug, should be removed later
constexpr auto kMaskedIntermCbIndex = tt::CBIndex::c_22;

constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kSingleTileBuffer = 1U;
constexpr uint32_t kNumOfIntermCBTiles = 2U;

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kFP32DestAccEnKey = "FP32_DEST_ACC_EN";

}  // namespace

namespace ttml::metal::ops::sdpa_bw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct SDPABackwardKernels {
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
    const SDPABackwardKernels& kernels,
    const tt::tt_metal::Buffer* grad_output_buffer,
    const tt::tt_metal::Buffer* attn_output_buffer,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* mask_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    const tt::tt_metal::Buffer* grad_query_buffer,
    const tt::tt_metal::Buffer* grad_key_buffer,
    const tt::tt_metal::Buffer* grad_value_buffer,
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

        // Reader kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {grad_output_buffer->address(),
             attn_output_buffer->address(),
             query_buffer->address(),
             key_buffer->address(),
             value_buffer->address(),
             mask_buffer != nullptr ? mask_buffer->address() : 0U,
             intermediates_buffer->address(),
             num_rows_per_core,
             num_rows_written});

        // Writer kernel runtime args
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {grad_query_buffer->address(),
             grad_key_buffer->address(),
             grad_value_buffer->address(),
             num_rows_per_core,
             num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

SDPABackwardProgramFactory::cached_program_t SDPABackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
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
    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // Get tensor dimensions and extract heads from shapes
    auto [qB, qNH, qS, qEmbd] = grad_output.padded_shape().to_array_4D();
    auto [kB, kNH, kS, kEmbd] = key.padded_shape().to_array_4D();
    auto [vB, vNH, vS, vEmbd] = value.padded_shape().to_array_4D();

    TT_FATAL(
        grad_output.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            query.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            key.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            value.physical_volume() % tt::constants::TILE_WIDTH == 0,
        "Physical volume of input tensors must be multiple of TILE_WIDTH. Got grad_output {}, query {}, key {}, value "
        "{}",
        grad_output.physical_volume(),
        query.physical_volume(),
        key.physical_volume(),
        value.physical_volume());

    TT_FATAL(qEmbd == kEmbd && qEmbd == vEmbd, "Embedding dims of grad_output, Q, K, V must be the same");
    TT_FATAL(qB == kB && qB == vB, "Query,Key and Value batch sizes must be the same");
    TT_FATAL(qS == kS && qS == vS, "Query, Key and Value sequence lengths must be the same");
    TT_FATAL(kNH == vNH, "Key and Value number of heads must be the same");

    // For backward pass we split work over rows of K and V
    // Each row corresponds to a group in K and V, and all associated heads in Q
    // TODO[improvement](vmelnykov): explore splitting work over cores using assumption that attn_mask is
    // causal(triangular).
    uint32_t St = qS / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    uint32_t NC = kB * kNH;
    uint32_t total_rows_to_process = NC * St;  // total rows to process = batch_size * num_heads * num_tiles_in_seq_len
    uint32_t kv_heads = kNH;                   // number of heads in Key and Value

    TT_FATAL(
        qNH % kv_heads == 0,
        "Number of heads must be divisible by number of groups, got heads={}, groups={}",
        qNH,
        kv_heads);

    uint32_t heads_per_group = qNH / kv_heads;         // we read heads_per_group heads from Q for one group of K and V
    uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;

    TT_FATAL(qWt == kWt && qWt == vWt, "Query, Key and Value inner dims must be the same");

    // Scale factor for attention computation
    float per_head_dim = static_cast<float>(qEmbd) / static_cast<float>(qNH);
    uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(per_head_dim));
    uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    uint32_t custom_inf = std::bit_cast<uint32_t>(1e9F);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t block_size = get_block_size(qWt, 4U);

    //[DEBUG]:
    fmt::print(
        "SDPA BW: NC={}, St={}, qWt={}, scaler = {}, block_size={}, q_heads = {}, kv_heads = {}, heads_per_group = "
        "{}, total_rows_to_process = {}, num_cores={} ({}x{}), "
        "group1 cores={} rows/core={}, group2 cores={} rows/core={}\n",
        NC,
        St,
        qWt,
        scaler,
        block_size,
        qNH,
        kv_heads,
        heads_per_group,
        total_rows_to_process,
        num_cores,
        num_cores_x,
        num_cores_y,
        core_group_1.size(),
        num_rows_per_core_group_1,
        core_group_2.size(),
        num_rows_per_core_group_2);

    auto data_format = input_data_format;
    auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_grad_output = create_circular_buffer(
        program, all_cores, kGradOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_attn_output = create_circular_buffer(
        program, all_cores, kAttnOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_query = create_circular_buffer(
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_key =
        create_circular_buffer(program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt);

    [[maybe_unused]] auto cb_value = create_circular_buffer(
        program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt);

    [[maybe_unused]] auto cb_attn_mask = create_circular_buffer(
        program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    // Could we write intermediates in fp32 to improve numerical stability?
    [[maybe_unused]] auto cb_intermediates = create_circular_buffer(
        program, all_cores, kIntermediatesCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumOfIntermCBTiles);

    // Utility buffers
    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);
    
    [[maybe_unused]] auto cb_prev_grad_value_holder = create_circular_buffer(
        program, all_cores, kPrevGradValueHolderCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_cur_grad_value_holder = create_circular_buffer(
        program, all_cores, kCurGradValueHolderCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_prev_grad_key_holder = create_circular_buffer(
        program, all_cores, kPrevGradKeyHolderCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_cur_grad_key_holder = create_circular_buffer(
        program, all_cores, kCurGradKeyHolderCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    // used to hold transpose value of attention weights, which is used in grad_V computation
    [[maybe_unused]] auto cb_transpose_wh = create_circular_buffer(
        program, all_cores, kTransposeWhCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    // Intermediate computation buffers
    [[maybe_unused]] auto cb_attention_weights = create_circular_buffer(
        program, all_cores, kAttentionWeightsCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_attention = create_circular_buffer(
        program, all_cores, kGradAttentionCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_scores = create_circular_buffer(
        program, all_cores, kGradScoresCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_u_scaler_row = create_circular_buffer(
        program, all_cores, kUScalarRowCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    // Output gradient buffers
    [[maybe_unused]] auto cb_grad_query = create_circular_buffer(
        program, all_cores, kGradQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_grad_key =
        create_circular_buffer(program, all_cores, kGradKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, kWt);

    [[maybe_unused]] auto cb_grad_value = create_circular_buffer(
        program, all_cores, kGradValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kWt);

    [[maybe_unused]] auto cb_sync_output_writer = create_circular_buffer(
        program, all_cores, kSyncOutputWriterCbIndex, data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    // [DEBUG]: Used for debug, should be removed later
    [[maybe_unused]] auto cb_masked_interm = create_circular_buffer(
        program, all_cores, kMaskedIntermCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumOfIntermCBTiles);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* grad_output_buffer = grad_output.buffer();
    auto* attn_output_buffer = attn_output.buffer();
    auto* query_buffer = query.buffer();
    auto* key_buffer = key.buffer();
    auto* value_buffer = value.buffer();
    auto* mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask.value().buffer() : nullptr;
    auto* intermediates_buffer = intermediates.buffer();

    auto* grad_query_buffer = output[0].buffer();
    auto* grad_key_buffer = output[1].buffer();
    auto* grad_value_buffer = output[2].buffer();

    // Configure defines
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    if (args.fp32_dest_acc_en) {
        defines[kFP32DestAccEnKey] = "1";
    }

    SDPABackwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // query width in tiles
        kWt,              // key/value width in tiles
        St,               // sequence length in tiles
        block_size,       // block size
        qNH,              // number of query heads
        heads_per_group,  // heads per group
        qB,               // num of batches
        scaler,           // sqrt(Et) - sdpa scale factor
        minus_one,        // used to transform mask from 1/0 to 0/-1
        custom_inf        // used to transform mask from 0/-1 to 0/-1e9F
    };
    tt::tt_metal::TensorAccessorArgs(grad_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(attn_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, defines, kReaderKernelPath);

    // Writer compile-time arguments
    std::vector<uint32_t> writer_compile_args = {
        qWt,             // query width in tiles
        kWt,             // key/value width in tiles
        St,              // sequence length in tiles
        block_size,      // block size
        qNH,             // number of query heads
        heads_per_group  // heads per group
    };
    tt::tt_metal::TensorAccessorArgs(grad_query_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(grad_key_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(grad_value_buffer).append_to(writer_compile_args);

    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        qWt,                        // query width in tiles
        kWt,                        // key/value width in tiles
        St,                         // sequence length in tiles
        qNH,                        // number of query heads
        heads_per_group,            // heads per group
        scaler,                     // scale factor
        minus_one,                  // mask transform constant
        custom_inf                  // mask transform constant
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, args.fp32_dest_acc_en);

    // Group 2 (if present)
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            qWt,                        // query width in tiles
            kWt,                        // key/value width in tiles
            St,                         // sequence length in tiles
            qNH,                        // number of query heads
            heads_per_group,            // heads per group
            scaler,                     // scale factor
            minus_one,                  // mask transform constant
            custom_inf                  // mask transform constant
        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, args.fp32_dest_acc_en);
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
        mask_buffer,
        intermediates_buffer,
        grad_query_buffer,
        grad_key_buffer,
        grad_value_buffer,
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
        {/* sdpa_bw_reader_kernel  = */ kernels.reader,
         /* sdpa_bw_writer_kernel  = */ kernels.writer,
         /* sdpa_bw_kernel_group_1 = */ kernels.compute_group_1,
         /* sdpa_bw_kernel_group_2 = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void SDPABackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared_vars = cached_program.shared_variables;
    auto& sdpa_bw_reader_kernel = shared_vars.sdpa_bw_reader_kernel;
    auto& sdpa_bw_writer_kernel = shared_vars.sdpa_bw_writer_kernel;
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

    auto* grad_query_buffer = tensor_return_value[0].buffer();
    auto* grad_key_buffer = tensor_return_value[1].buffer();
    auto* grad_value_buffer = tensor_return_value[2].buffer();

    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_bw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_bw_writer_kernel);

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kGradOutputBufferIdx] = grad_output_buffer->address();
            runtime_args[kAttnOutputBufferIdx] = attn_output_buffer->address();
            runtime_args[kQueryBufferIdx] = query_buffer->address();
            runtime_args[kKeyBufferIdx] = key_buffer->address();
            runtime_args[kValueBufferIdx] = value_buffer->address();
            runtime_args[kAttnMaskBufferIdx] = mask_buffer != nullptr ? mask_buffer->address() : 0;
            runtime_args[kIntermediatesBufferIdx] = intermediates_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kGradQueryBufferIdx] = grad_query_buffer->address();
            runtime_args[kGradKeyBufferIdx] = grad_key_buffer->address();
            runtime_args[kGradValueBufferIdx] = grad_value_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::sdpa_bw::device
