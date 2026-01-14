// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_writer_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp";

// reader runtime args
constexpr uint32_t kQueryBufferIdx = 0;
constexpr uint32_t kKeyBufferIdx = 1U;
constexpr uint32_t kValueBufferIdx = 2U;
constexpr uint32_t kMaskBufferIdx = 3U;

// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;
constexpr uint32_t kIntermediateBufferIdx = 1U;

constexpr auto kQueryCbIndex = tt::CBIndex::c_0;
constexpr auto kKeyCbIndex = tt::CBIndex::c_1;
constexpr auto kValueCbIndex = tt::CBIndex::c_2;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_3;
constexpr auto kIntermediateCbIndex = tt::CBIndex::c_4;
constexpr auto kReductionScalerCbIndex = tt::CBIndex::c_5;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;  // used for matmul reduction
constexpr auto kQKResultCbIndex = tt::CBIndex::c_7;      // used for accumulating results

constexpr auto kPrevMaxValueCbIndex = tt::CBIndex::c_8;  // used for holding max value during reduce
constexpr auto kCurMaxValueCbIndex = tt::CBIndex::c_9;   // used for holding max value during reduce
constexpr auto kExpMaxDiffCbIndex = tt::CBIndex::c_10;   // used for holding exp sum diff during reduce
constexpr auto kPrevSumExpCbIndex = tt::CBIndex::c_11;   // used for holding exp sum during reduce
constexpr auto kCurSumExpCbIndex = tt::CBIndex::c_12;    // used for holding exp sum during reduce
constexpr auto kPrevMmOutCbIndex = tt::CBIndex::c_13;    // used for holding previous matmul output
constexpr auto kCurMmOutCbIndex = tt::CBIndex::c_14;     // used for holding current matmul output

constexpr auto kOutputCbIndex = tt::CBIndex::c_15;

constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumAttnMaskTiles = 1U;
constexpr uint32_t kQKResultTiles = 1U;
constexpr uint32_t kMaxValueHolderTiles = 1U;
constexpr uint32_t kExpMaxDiffTiles = 1U;
constexpr uint32_t kExpSumTiles = 1U;
constexpr uint32_t kIntermediateTiles = 2U;  // max_val at col 0, recip_sum_exp at col 32

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";

}  // namespace

namespace ttml::metal::ops::sdpa_fw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct SDPAForwardKernels {
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
    const SDPAForwardKernels& kernels,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* mask_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
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

        // Reader kernel: (query_addr, key_addr, value_addr, mask_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {query_buffer->address(),
             key_buffer->address(),
             value_buffer->address(),
             mask_buffer != nullptr ? mask_buffer->address() : 0,
             num_rows_per_core,
             num_rows_written});

        // Writer kernel: (output_addr, intermediates_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(),
             intermediates_buffer != nullptr ? intermediates_buffer->address() : 0,
             num_rows_per_core,
             num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

SDPAForwardProgramFactory::cached_program_t SDPAForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& attn_mask = tensor_args.mask;
    /*
    Shape note:
    Q: B x qNH x S x qE
    K: B x kNH x S x kE
    V: B x vNH x S x vE
    attn_mask: B x qNH x S x S
    */

    auto* device = query.device();

    tt::tt_metal::Program program{};
    auto input_data_format = datatype_to_dataformat_converter(query.dtype());
    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto [qB, qNH, qS, qEmbd] = query.padded_shape().to_array_4D();
    auto [kB, kNH, kS, kEmbd] = key.padded_shape().to_array_4D();
    auto [vB, vNH, vS, vEmbd] = value.padded_shape().to_array_4D();
    TT_FATAL(
        query.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            key.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            value.physical_volume() % tt::constants::TILE_WIDTH == 0,
        "Physical volume of input tensors must be multiple of TILE_WIDTH. Got query {}, key {}, value {}",
        query.physical_volume(),
        key.physical_volume(),
        value.physical_volume());

    TT_FATAL(qEmbd == kEmbd && qEmbd == vEmbd, "Embedding dims of Q, K, V must be the same");
    TT_FATAL(qB == kB, "Query and Key batch sizes must be the same");
    TT_FATAL(qS == kS, "Query and Key sequence lengths must be the same");

    uint32_t St = qS / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    uint32_t NC = qB * qNH;
    uint32_t total_rows_to_process = NC * St;  // total rows to process = batch_size * num_heads * num_tiles_in_seq_len

    TT_FATAL(kNH == vNH, "Number of heads in Key and Value must be the same");
    uint32_t kv_heads = kNH;  // number of heads in Key and Value

    TT_FATAL(
        qNH % kv_heads == 0,
        "Number of heads must be divisible by number of groups, got heads={}, groups={}",
        qNH,
        kv_heads);

    uint32_t heads_per_group = qNH / kv_heads;         // we read heads_per_group heads from Q for one group of K and V
    uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;

    uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(qEmbd)));  // calculate scale factor
    uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);  // used to transform mask from 1/0 to 0/-1
    uint32_t custom_inf = std::bit_cast<uint32_t>(1e9F);  // used to transform mask from 0/-1 to 0/-1e9F

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    /* TODO[optimization](vmelnykov): #29160 - explore more efficient ways to split work across kernels.
     * For example, instead of processing a single row per core, process multiple rows at once
     * (e.g., q_chunks_size = 2). This allows better utilization of available cores, improves
     * matmul efficiency (subblock 2x4), and amortizes init/acquire/release overhead across rows.
     *
     */
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t block_size = get_block_size(qWt, 4U);

    const bool use_attn_mask = attn_mask.has_value();

    auto data_format = input_data_format;
    auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_query = create_circular_buffer(
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_key =
        create_circular_buffer(program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt);

    [[maybe_unused]] auto cb_value = create_circular_buffer(
        program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

    // create mask buffer only if it's going to be used
    if (use_attn_mask) {
        [[maybe_unused]] auto cb_attn_mask = create_circular_buffer(
            program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumAttnMaskTiles);
    }

    // create intermediate buffer only if we need to return intermediates
    // Intermediate shape: (B, H, S, 64) = 2 tiles wide (max_val at col 0, recip_sum_exp at col 32)
    if (args.return_intermediates) {
        [[maybe_unused]] auto cb_intermediate = create_circular_buffer(
            program, all_cores, kIntermediateCbIndex, data_format, bfloat16_single_tile_size_bytes, kIntermediateTiles);
    }

    [[maybe_unused]] auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    [[maybe_unused]] auto cb_qk_result = create_circular_buffer(
        program, all_cores, kQKResultCbIndex, data_format, bfloat16_single_tile_size_bytes, kQKResultTiles);

    [[maybe_unused]] auto cb_prev_max_value = create_circular_buffer(
        program, all_cores, kPrevMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kMaxValueHolderTiles);

    [[maybe_unused]] auto cb_cur_max_value = create_circular_buffer(
        program, all_cores, kCurMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kMaxValueHolderTiles);

    // lets try to use precise data format for holding exp sum/diff values
    [[maybe_unused]] auto cb_exp_max_diff = create_circular_buffer(
        program, all_cores, kExpMaxDiffCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpMaxDiffTiles);

    [[maybe_unused]] auto cb_prev_exp_sum = create_circular_buffer(
        program, all_cores, kPrevSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpSumTiles);

    [[maybe_unused]] auto cb_cur_exp_sum = create_circular_buffer(
        program, all_cores, kCurSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpSumTiles);

    [[maybe_unused]] auto cb_prev_mm_out = create_circular_buffer(
        program, all_cores, kPrevMmOutCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_cur_mm_out =
        create_circular_buffer(program, all_cores, kCurMmOutCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, qWt);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* query_buffer = query.buffer();
    TT_FATAL(
        query_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Query buffer must be in DRAM. Query buffer of type {}",
        enchantum::to_string(query_buffer->buffer_type()));

    auto* key_buffer = key.buffer();
    TT_FATAL(
        key_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Key buffer must be in DRAM. Key buffer of type {}",
        enchantum::to_string(key_buffer->buffer_type()));

    auto* value_buffer = value.buffer();
    TT_FATAL(
        value_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Value buffer must be in DRAM. Value buffer of type {}",
        enchantum::to_string(value_buffer->buffer_type()));

    auto* mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;
    if (mask_buffer != nullptr) {
        TT_FATAL(
            mask_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Mask buffer must be in DRAM. Mask buffer of type {}",
            enchantum::to_string(mask_buffer->buffer_type()));
    }

    auto* output_buffer = output.front().buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    auto* intermediates_buffer = args.return_intermediates ? output.back().buffer() : nullptr;
    if (intermediates_buffer != nullptr) {
        TT_FATAL(
            intermediates_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Intermediates buffer must be in DRAM. Intermediates buffer of type {}",
            enchantum::to_string(intermediates_buffer->buffer_type()));
    }

    // configure defines
    std::map<std::string, std::string> defines;
    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    if (args.return_intermediates) {
        defines[kReturnIntermediates] = "1";
    }

    if (use_attn_mask) {
        defines[kUseAttnMaskDefKey] = "1";
    }

    SDPAForwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // num tile in inner dim in query(d/TILE_W)
        kWt,              // num tile in inner dim in key and value (d/TILE_W)
        St,               // num tile in seq len dim (S/TILE_H)
        block_size,       // block size (dst_reg_count)
        qNH,              // number of heads in query
        heads_per_group,  // number of heads per group
        qB,               // num of batches
        scaler,           // sqrt(Et) - sdpa scale factor
        minus_one,        // used to transform mask from 1/0 to 0/-1
        custom_inf        // used to transform mask from 0/-1 to 0/-1e9F
    };
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ reader_compile_args,
        defines,
        kReaderKernelPath);

    std::vector<uint32_t> writer_compile_args = {
        qWt,             // num tile in inner dim in query(d/TILE_W)
        St,              // num tile in seq len dim (S/TILE_H)
        block_size,      // block size (dst_reg_count)
        qNH,             // number of heads in query
        heads_per_group  // number of heads per group
    };
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(writer_compile_args);
    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ writer_compile_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for rmsnorm_fw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        qWt,                        // num tile in inner dim in query(d/TILE_W)
        kWt,                        // num tile in inner dim in key and value (d/TILE_W)
        St,                         // num_seq_len / TILE_H
        qNH,                        // number of heads in query
        heads_per_group,            // number of heads per group
        scaler,                     // sqrt(Et) - sdpa scaler factor
        minus_one,                  // used to transform mask from 1/0 to 0/-1
        custom_inf                  // used to transform mask from 0/-1 to 0/-1e9F
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /* fp32_dest_acc_en */ true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            qWt,                        // num tile in inner dim in query(d/TILE_W)
            kWt,                        // num tile in inner dim in key and value (d/TILE_W)
            St,                         // num_seq_len / TILE_H
            qNH,                        // number of heads in query
            heads_per_group,            // number of heads per group
            scaler,                     // sqrt(Et) - sdpa scaler factor
            minus_one,                  // used to transform mask from 1/0 to 0/-1
            custom_inf                  // used to transform mask from 0/-1 to 0/-1e9F
        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, /* fp32_dest_acc_en */ true);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        query_buffer,
        key_buffer,
        value_buffer,
        mask_buffer,
        output_buffer,
        intermediates_buffer,
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
        {/* sdpa_fw_reader_kernel  = */ kernels.reader,
         /* sdpa_fw_writer_kernel  = */ kernels.writer,
         /* sdpa_fw_kernel_group_1 = */ kernels.compute_group_1,
         /* sdpa_fw_kernel_group_2 = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void SDPAForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared_vars = cached_program.shared_variables;
    auto& sdpa_fw_reader_kernel = shared_vars.sdpa_fw_reader_kernel;
    auto& sdpa_fw_writer_kernel = shared_vars.sdpa_fw_writer_kernel;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto* query_buffer = tensor_args.query.buffer();
    const auto* key_buffer = tensor_args.key.buffer();
    const auto* value_buffer = tensor_args.value.buffer();
    const auto* mask_buffer = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer() : nullptr;
    auto* output_buffer = tensor_return_value.front().buffer();
    auto* intermediates_buffer =
        operation_attributes.return_intermediates ? tensor_return_value.back().buffer() : nullptr;

    // Only address arguments need updating here; tile counts remain the same as in create().
    // No runtime args to update for compute kernels.
    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_fw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_fw_writer_kernel);

    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kQueryBufferIdx] = query_buffer->address();
            runtime_args[kKeyBufferIdx] = key_buffer->address();
            runtime_args[kValueBufferIdx] = value_buffer->address();
            runtime_args[kMaskBufferIdx] = mask_buffer != nullptr ? mask_buffer->address() : 0;
        }

        // Update output buffer for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
            runtime_args[kIntermediateBufferIdx] =
                intermediates_buffer != nullptr ? intermediates_buffer->address() : 0;
        }
    }
}

}  // namespace ttml::metal::ops::sdpa_fw::device
