// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <bit>
#include <cmath>

#include "metal/ops/common/program_utils.hpp"

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
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_6;  // used for transposing key tiles
constexpr auto kTempAccumCbIndex = tt::CBIndex::c_7;     // used for accumulating results

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
constexpr uint32_t kQKResultTiles = 1U;  //[Debug] should be 2U
constexpr uint32_t kMaxValueHolderTiles = 1U;
constexpr uint32_t kExpMaxDiffTiles = 1U;
constexpr uint32_t kExpSumTiles = 1U;
constexpr uint32_t kIntermediateTiles = 1U;  // [Debug] should be 2U
constexpr uint32_t kOnetile = 1U;

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";

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
             mask_buffer != nullptr ? mask_buffer->address() : 0U,
             num_rows_per_core,
             num_rows_written});

        // Writer kernel: (output_addr, intermediates_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(), intermediates_buffer->address(), num_rows_per_core, num_rows_written});

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
    Q: B x 1U x S x E
    K: B x 1U x S x E
    V: B x 1U x S x E
    attn_mask: B x 1U x S x S
    */

    auto* device = query.device();

    tt::tt_metal::Program program{};
    auto input_data_format = datatype_to_dataformat_converter(query.dtype());
    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);

    /*
     * Split embedding dim into heads and groups
     * Two cases:
     * 1) H_q == H_k == H_v == G: each head has its own K and V:
     *    For this case we read and process data by heads: Edim/H_q
     * 2) H_q == n * G, n > 1: each group of K and V is shared across n heads
     *    For this case we read and process data by heads: Edim/H_q
     */

    // [Debug] could I assume that Wt%(heads*TILE_WIDTH) == 0 ?
    auto [qBt, qHt, qSt, qDt] = query.padded_shape().to_array_4D();
    auto [kBt, kHt, kSt, kDt] = key.padded_shape().to_array_4D();
    auto [vBt, vHt, vSt, vDt] = value.padded_shape().to_array_4D();
    TT_FATAL(
        query.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            key.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            value.physical_volume() % tt::constants::TILE_WIDTH == 0,
        "Physical volume of input tensors must be multiple of TILE_WIDTH. Got query {}, key {}, value {}",
        query.physical_volume(),
        key.physical_volume(),
        value.physical_volume());

    uint32_t Wt = qDt / tt::constants::TILE_WIDTH;    // num of tiles in inner dim
    uint32_t Ht_ = qSt / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    uint32_t NC = qBt * qHt;
    uint32_t total_rows_to_process = NC * Ht_;

    uint32_t q_heads = args.q_heads;    // will be passed by user into args
    uint32_t kv_heads = args.kv_heads;  // will be passed by user into args
    TT_FATAL(
        q_heads % kv_heads == 0,
        "Number of heads must be divisible by number of groups, got heads={}, groups={}",
        q_heads,
        kv_heads);

    TT_FATAL(qBt == kBt, "Query and Key batch sizes must be the same");
    TT_FATAL(qSt == kSt, "Query and Key sequence lengths must be the same");

    uint32_t heads_per_group = q_heads / kv_heads;   // we read heads_per_group heads from Q for one group of K and V
    uint32_t qWt = qDt / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    uint32_t kWt = kDt / tt::constants::TILE_WIDTH;
    uint32_t vWt = vDt / tt::constants::TILE_WIDTH;
    uint32_t q_tiles_per_head = qWt / q_heads;   // number of tiles per head in query
    uint32_t k_tiles_per_head = kWt / kv_heads;  // number of tiles per group in key
    uint32_t v_tiles_per_head = vWt / kv_heads;  // number of tiles per group in value

    TT_FATAL(
        q_tiles_per_head == k_tiles_per_head && q_tiles_per_head == v_tiles_per_head,
        "Number of tiles per head in Query, Key, and Value must be the same");

    float per_head_dim = static_cast<float>(qDt) / static_cast<float>(q_heads);
    uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(per_head_dim));  // calculate scale factor
    uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);  // used to transform mask from 1/0 to 0/-1
    uint32_t custom_inf = std::bit_cast<uint32_t>(1e9F);  // used to transform mask from 0/-1 to 0/-1e9F

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    /* TODO[improve]: think about how to split work around kernels more efficiently
     * For example, if we have 8 cores and 4 rows with two heads each (total 8 heads),
     * we can use 4 cores to process 4 rows in parallel (one head per core) and then use the other 4 cores to process
     * the same 4 rows in parallel (the second head per core) This way we can utilize all 8 cores and reduce the overall
     * processing time
     */
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    uint32_t block_size = get_block_size(Wt, 4U);
    uint32_t twice_block_size = 2 * block_size;

    //[DEBUG]:
    fmt::print(
        "SDPA FW: NC={}, Ht_={}, Wt={}, scaler = {}, block_size={}, q_heads = {}, kv_heads = {}, total_rows_to_process "
        "= {}, num_cores={} ({}x{}), "
        "group1 cores={} rows/core={}, group2 "
        "cores={} "
        "rows/core={}\n",
        NC,
        Ht_,
        Wt,
        scaler,
        block_size,
        q_heads,
        kv_heads,
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

    auto cb_query =
        create_circular_buffer(program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * Wt);

    auto cb_key =
        create_circular_buffer(program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * Wt);

    auto cb_value =
        create_circular_buffer(program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * Wt);

    auto cb_attn_mask = create_circular_buffer(
        program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumAttnMaskTiles);

    auto cb_intermediate = create_circular_buffer(
        program, all_cores, kIntermediateCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_temp_accum = create_circular_buffer(
        program, all_cores, kTempAccumCbIndex, data_format, bfloat16_single_tile_size_bytes, kQKResultTiles);

    auto cb_prev_max_value = create_circular_buffer(
        program, all_cores, kPrevMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kMaxValueHolderTiles);

    auto cb_cur_max_value = create_circular_buffer(
        program, all_cores, kCurMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kMaxValueHolderTiles);

    // lets try to use precise data format for holding exp sum/diff values
    auto cb_exp_max_diff = create_circular_buffer(
        program, all_cores, kExpMaxDiffCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpMaxDiffTiles);

    auto cb_prev_exp_sum = create_circular_buffer(
        program, all_cores, kPrevSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpSumTiles);

    auto cb_cur_exp_sum = create_circular_buffer(
        program, all_cores, kCurSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, kExpSumTiles);

    auto cb_prev_mm_out =
        create_circular_buffer(program, all_cores, kPrevMmOutCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    auto cb_cur_mm_out =
        create_circular_buffer(program, all_cores, kCurMmOutCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    auto cb_mm_result_holder =
        create_circular_buffer(program, all_cores, tt::CBIndex::c_16, data_format, bfloat16_single_tile_size_bytes, Wt);

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

    auto* intermediates_buffer = output.back().buffer();
    TT_FATAL(
        intermediates_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Intermediates buffer must be in DRAM. Intermediates buffer of type {}",
        enchantum::to_string(intermediates_buffer->buffer_type()));

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

    SDPAForwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,               // num tile in inner dim in query(d/TILE_W)
        kWt,               // num tile in inner dim in key and value (d/TILE_W)
        Ht_,               // num tile in seq len dim (S/TILE_H)
        block_size,        // block size (dst_reg_count)
        q_tiles_per_head,  // number of tiles per head in query
        q_heads,           // number of heads in query
        k_tiles_per_head,  // number of tiles per group in key and value
        heads_per_group,   // number of heads per group
        scaler,            // sqrt(Et) - sdpa scale factor
        minus_one,         // used to transform mask from 1/0 to 0/-1
        custom_inf,        // used to transform mask from 0/-1 to 0/-1e9F
        Wt                 // old Wt[used only for debug]
    };
    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ reader_compile_args,
        defines,
        kReaderKernelPath);

    std::vector<uint32_t> writer_compile_args = {
        qWt,               // num tile in inner dim in query(d/TILE_W)
        Ht_,               // num tile in seq len dim (S/TILE_H)
        block_size,        // block size (dst_reg_count)
        q_tiles_per_head,  // number of tiles per head in query
        q_heads,           // number of heads in query
        heads_per_group,
        Wt  // old Wt[used only for debug]
    };
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
        Ht_,                        // num_seq_len / TILE_H
        q_tiles_per_head,           // number of tiles per head in query
        q_heads,                    // number of heads in query
        k_tiles_per_head,           // number of tiles per group in key and value
        heads_per_group,            // number of heads per group
        scaler,                     // sqrt(Et) - sdpa scaler factor
        minus_one,                  // used to transform mask from 1/0 to 0/-1
        custom_inf,                 // used to transform mask from 0/-1 to 0/-1e9F
        Wt                          // old Wt[used only for debug]
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
            Ht_,                        // num_seq_len / TILE_H
            q_tiles_per_head,           // number of tiles per head in query
            q_heads,                    // number of heads in query
            k_tiles_per_head,           // number of tiles per group in key and value
            heads_per_group,            // number of heads per group
            scaler,                     // sqrt(Et) - sdpa scaler factor
            minus_one,                  // used to transform mask from 1/0 to 0/-1
            custom_inf,                 // used to transform mask from 0/-1 to 0/-1e9F
            Wt                          // old Wt[used only for debug]
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
    auto& sdpa_fw_group_1_kernel = shared_vars.sdpa_fw_kernel_group_1;
    auto& sdpa_fw_group_2_kernel = shared_vars.sdpa_fw_kernel_group_2;
    auto& core_group_1 = shared_vars.core_group_1;
    auto& core_group_2 = shared_vars.core_group_2;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto* query_buffer = tensor_args.query.buffer();
    const auto* key_buffer = tensor_args.key.buffer();
    const auto* value_buffer = tensor_args.value.buffer();
    const auto* mask_buffer = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer() : nullptr;
    auto* output_buffer = tensor_return_value.front().buffer();
    auto* intermediates_buffer = tensor_return_value.back().buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_fw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_fw_writer_kernel);
    // auto& group_1_runtime_args = GetRuntimeArgs(program, sdpa_fw_group_1_kernel);
    // // we need to initialize it with something, but if group 2 is  empty it will be used in the loop
    // auto& group_2_runtime_args =
    //     core_group_2.ranges().empty() ? group_1_runtime_args : GetRuntimeArgs(program, sdpa_fw_group_2_kernel);

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

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
            runtime_args[kIntermediateBufferIdx] = intermediates_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::sdpa_fw::device
