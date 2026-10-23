// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>

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
constexpr uint32_t kGateBufferIdx = 4U;

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
constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_7;

constexpr auto kPrevMaxValueCbIndex = tt::CBIndex::c_8;  // used for holding max value during reduce
constexpr auto kCurMaxValueCbIndex = tt::CBIndex::c_9;   // used for holding max value during reduce
constexpr auto kExpMaxDiffCbIndex = tt::CBIndex::c_10;   // used for holding exp sum diff during reduce
constexpr auto kPrevSumExpCbIndex = tt::CBIndex::c_11;   // used for holding exp sum during reduce
constexpr auto kCurSumExpCbIndex = tt::CBIndex::c_12;    // used for holding exp sum during reduce
constexpr auto kPrevMmOutCbIndex = tt::CBIndex::c_13;    // used for holding previous matmul output
constexpr auto kCurMmOutCbIndex = tt::CBIndex::c_14;     // used for holding current matmul output

constexpr auto kOutputCbIndex = tt::CBIndex::c_15;
constexpr auto kGateCbIndex = tt::CBIndex::c_16;

constexpr uint32_t kNumScalerTiles = 1U;
constexpr uint32_t kNumCausalMaskTiles = 2U;  // [0] = causal-diag pattern (0 kept / -1e9 masked),
                                              // [1] = all -1e9 (strictly past the diagonal)
constexpr uint32_t kIntermediateTiles = 1U;   // logsumexp = max + log(sum_exp), single FP32 tile

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";

const std::string kHasGateDefKey = "HAS_GATE";

// Pick the multi-tile K/V chunk size. Returns the number of K tiles processed per inner-loop
// iteration of the compute kernel.
//
// Constraints:
//   1. Must divide Ht (so the diagonal of the causal mask falls cleanly within a chunk,
//      and so the reader never reads past the end of the sequence).
//   2. Must fit DST under fp32_dest_acc_en=true (DST has 4 tiles): the chunked softmax
//      holds Sk_chunk_t exp tiles live across one tile_regs cycle, so we cap at 4.
//   3. Larger chunks amortize softmax / correction overhead more, but consume L1
//      proportionally for cb_key, cb_value, and cb_attention_weights.
//
// We pick the largest power-of-2 in {1, 2, 4} that divides Ht. Bumping past 4 would require
// either fp32_dest_acc_en=false (loses precision) or splitting the exp pass into multiple
// DST cycles.
uint32_t pick_sk_chunk_t(uint32_t Ht) {
    constexpr uint32_t kMaxSkChunkT = 4U;
    for (uint32_t c = kMaxSkChunkT; c > 1U; c /= 2U) {
        if (Ht % c == 0U) {
            return c;
        }
    }
    return 1U;
}

// Chunk knob in elements -> tiles. 0 = fallback; otherwise it must be a tile multiple that divides S.
// Checked here (not in validate) so the ring SDPA path, which builds the program directly, is covered too.
uint32_t chunk_size_to_tiles(uint32_t chunk_size, uint32_t S, uint32_t fallback, const char* name) {
    if (chunk_size == 0U) {
        return fallback;
    }
    TT_FATAL(
        chunk_size % tt::constants::TILE_HEIGHT == 0U && S % chunk_size == 0U,
        "{} ({}) must be a multiple of {} that divides the sequence length ({})",
        name,
        chunk_size,
        tt::constants::TILE_HEIGHT,
        S);
    return chunk_size / tt::constants::TILE_HEIGHT;
}

// One matmul_block call produces a rectangle of output tiles (block_rows x block_cols) that must fit
// in DST (dst_size tiles). The full output is rows x cols tiles, so the rectangle sides must divide
// those. Take the biggest rectangle; between equal sizes take the wider one, because the tiles of
// one output row are consecutive in memory and a wider rectangle re-reads the left operand less.
struct MatmulBlockDims {
    uint32_t block_rows;
    uint32_t block_cols;
};

MatmulBlockDims pick_matmul_block_dims(uint32_t rows, uint32_t cols, uint32_t dst_size) {
    MatmulBlockDims best{1U, 1U};
    for (uint32_t block_rows = 1U; block_rows <= rows; ++block_rows) {
        if (rows % block_rows != 0U) {
            continue;
        }
        for (uint32_t block_cols = 1U; block_cols <= cols; ++block_cols) {
            if (cols % block_cols != 0U || block_rows * block_cols > dst_size) {
                continue;
            }
            const uint32_t tiles = block_rows * block_cols;
            const uint32_t best_tiles = best.block_rows * best.block_cols;
            if (tiles > best_tiles || (tiles == best_tiles && block_cols > best.block_cols)) {
                best = {block_rows, block_cols};
            }
        }
    }
    return best;
}

// Work item = one Q chunk (Sq_chunk_t query tile-rows of one batch/head), indexed
// (batch * q_heads + head) * q_num_chunks + chunk. Each core gets a list of contiguous item runs and the
// three kernels walk that list in order; the list is the whole work-distribution contract, so a
// different policy only has to produce different lists.
struct ItemRun {
    uint32_t first_item;
    uint32_t num_items;
};
using CoreItemRuns = std::vector<ItemRun>;

void append_item(CoreItemRuns& runs, uint32_t item) {
    if (!runs.empty() && runs.back().first_item + runs.back().num_items == item) {
        ++runs.back().num_items;
    } else {
        runs.push_back({item, 1U});
    }
}

// One contiguous block of items per core; the first (total_items % num_cores) cores get one more.
std::vector<CoreItemRuns> split_items_contiguous(uint32_t total_items, uint32_t num_cores) {
    TT_FATAL(num_cores > 0, "split_items_contiguous: num_cores must be > 0");
    std::vector<CoreItemRuns> per_core(num_cores);
    const uint32_t items_per_core = total_items / num_cores;
    const uint32_t remainder = total_items % num_cores;
    uint32_t item = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const uint32_t num_items = items_per_core + (core < remainder ? 1U : 0U);
        per_core[core].push_back({item, num_items});
        item += num_items;
    }
    return per_core;
}

// Causal: Q chunk i of a sequence attends ~(i+1) K/V chunks, so pairing chunk i with chunk
// (q_num_chunks - 1 - i) gives every pair the same work. The pairs are handed out contiguously like
// items above; on each core the heavy chunks come first, then the light ones, both ascending so they
// fold into few runs.
std::vector<CoreItemRuns> split_items_paired(uint32_t num_seqs, uint32_t q_num_chunks, uint32_t num_cores) {
    const uint32_t pairs_per_seq = q_num_chunks / 2U;
    const auto pair_runs = split_items_contiguous(num_seqs * pairs_per_seq, num_cores);
    std::vector<CoreItemRuns> per_core(num_cores);
    for (uint32_t core = 0; core < num_cores; ++core) {
        const auto [first_pair, num_pairs] = pair_runs[core].front();
        for (uint32_t p = first_pair + num_pairs; p-- > first_pair;) {
            append_item(per_core[core], (p / pairs_per_seq) * q_num_chunks + (q_num_chunks - 1U - p % pairs_per_seq));
        }
        for (uint32_t p = first_pair; p < first_pair + num_pairs; ++p) {
            append_item(per_core[core], (p / pairs_per_seq) * q_num_chunks + p % pairs_per_seq);
        }
    }
    return per_core;
}

}  // namespace

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
};

// Runtime args: the buffer addresses (reader, writer), then the core's item runs for all three kernels:
// num_runs, (first_item, num_items) per run.
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SDPAForwardKernels& kernels,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* mask_buffer,
    const tt::tt_metal::Buffer* gate_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    uint32_t num_cores_y,
    const std::vector<CoreItemRuns>& per_core_runs) {
    constexpr uint32_t kReaderAddressArgs = 5U;
    for (uint32_t i = 0; i < per_core_runs.size(); i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& runs = per_core_runs[i];

        std::vector<uint32_t> work_args{static_cast<uint32_t>(runs.size())};
        for (const auto& run : runs) {
            work_args.push_back(run.first_item);
            work_args.push_back(run.num_items);
        }
        TT_FATAL(
            kReaderAddressArgs + work_args.size() <= tt::tt_metal::max_runtime_args,
            "SDPA forward: {} work runs on core {} exceed the runtime arg budget; use a larger q_chunk_size",
            runs.size(),
            core);

        std::vector<uint32_t> reader_args = {
            query_buffer->address(),
            key_buffer->address(),
            value_buffer->address(),
            mask_buffer != nullptr ? mask_buffer->address() : 0,
            gate_buffer != nullptr ? gate_buffer->address() : 0};
        reader_args.insert(reader_args.end(), work_args.begin(), work_args.end());
        SetRuntimeArgs(program, kernels.reader, core, reader_args);

        std::vector<uint32_t> writer_args = {
            output_buffer->address(), intermediates_buffer != nullptr ? intermediates_buffer->address() : 0};
        writer_args.insert(writer_args.end(), work_args.begin(), work_args.end());
        SetRuntimeArgs(program, kernels.writer, core, writer_args);

        SetRuntimeArgs(program, kernels.compute, core, work_args);
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
    const AttentionMaskType mask_type = args.mask_type;
    const auto& gate = tensor_args.gate;
    /*
    Shape note:
    Q: B x qNH x S x qE
    K: B x kNH x S x kE
    V: B x vNH x S x vE
    attn_mask: 1 x 1 x S x S
    */

    auto* device = query.device();

    tt::tt_metal::Program program{};
    const auto input_data_format = datatype_to_dataformat_converter(query.dtype());
    const uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

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

    const uint32_t St = qS / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    const uint32_t NC = qB * qNH;

    const uint32_t heads_per_group = qNH / kNH;  // we read heads_per_group heads from Q for one group of K and V
    const uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    const uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    const uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;

    // The two knobs in tiles (_t): Sq_chunk_t query tile-rows are processed together against one
    // pass over K/V, Sk_chunk_t K/V tiles are consumed per step. 0 = default (1 Q row, up to 4 K tiles).
    const uint32_t Sq_chunk_t =
        chunk_size_to_tiles(args.program_config.q_chunk_size, qS, /* fallback */ 1U, "q_chunk_size");
    const uint32_t Sk_chunk_t =
        chunk_size_to_tiles(args.program_config.k_chunk_size, qS, pick_sk_chunk_t(St), "k_chunk_size");

    const uint32_t q_num_chunks = St / Sq_chunk_t;
    const uint32_t total_q_chunks = NC * q_num_chunks;

    const float scale = 1.0F / std::sqrt(static_cast<float>(qEmbd));
    const uint32_t scaler = std::bit_cast<uint32_t>(scale);
    // BF16 copy of the scale for the WH fused scale+exp path (SFPU multiplies P by a BF16
    // immediate there, while the LSE and the backward pass use the full FP32 scale). RNE
    // instead of `>> 16` truncation halves the worst-case rounding gap so it stays noise
    // rather than a systematic fw/bw mismatch. The conversion runs on host because the
    // kernel TU cannot see the host bfloat16 implementation.
    const uint32_t scaler_bf16 = fp32_to_bf16_bits_round_to_nearest_even(scale);
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);  // used to transform mask from 1/0 to 0/-1
    const uint32_t custom_inf = std::bit_cast<uint32_t>(1e9F);  // used to transform mask from 0/-1 to 0/-1e9F

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_available_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // Causal with an even chunk count and enough pairs for every core: heavy/light chunk pairs.
    // Otherwise contiguous blocks of Q chunks.
    const uint32_t pairs_per_seq = q_num_chunks / 2;
    const uint32_t total_pairs = NC * pairs_per_seq;
    const bool use_paired_split = (mask_type == AttentionMaskType::Causal) && (q_num_chunks % 2 == 0) &&
                                  (total_pairs >= num_available_cores) && (St > 0);

    const uint32_t num_cores = std::min(num_available_cores, use_paired_split ? total_pairs : total_q_chunks);
    const auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size);
    const auto per_core_runs = use_paired_split ? split_items_paired(NC, q_num_chunks, num_cores)
                                                : split_items_contiguous(total_q_chunks, num_cores);

    // DST is the register file the math unit computes in: 4 tiles in fp32 mode (8 in bf16).
    // Everything below is "how many tiles go through DST per pass":
    //   block_size     output tiles per pass in the element-wise loops (one slot stays free for the
    //                  broadcast factor they multiply by)
    //   qk_block_*     Q rows x K tiles per QK^T matmul_block call
    //   pv_block_*     Q rows x V feature tiles per PV matmul_block call
    //   sm_block_size  score tiles of one row per pass of the exp step
    constexpr bool fp32_dest_acc_en = true;
    const uint32_t dst_size = fp32_dest_acc_en ? 4U : 8U;
    const uint32_t block_size = get_block_size(vWt, dst_size - 1U);
    const auto qk_block = pick_matmul_block_dims(Sq_chunk_t, Sk_chunk_t, dst_size);
    const uint32_t qk_block_h = qk_block.block_rows;
    const uint32_t qk_block_w = qk_block.block_cols;
    const auto pv_block = pick_matmul_block_dims(Sq_chunk_t, vWt, dst_size);
    const uint32_t pv_block_h = pv_block.block_rows;
    const uint32_t pv_block_w = pv_block.block_cols;
    const uint32_t sm_block_size = get_block_size(Sk_chunk_t, dst_size);

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    const uint32_t cb_query_tiles = 2U * Sq_chunk_t * qWt;
    const uint32_t cb_key_tiles = 2U * Sk_chunk_t * kWt;
    const uint32_t cb_value_tiles = 2U * Sk_chunk_t * vWt;
    const uint32_t cb_attn_mask_tiles =
        (mask_type == AttentionMaskType::Causal) ? kNumCausalMaskTiles : (2U * Sq_chunk_t * Sk_chunk_t);
    const uint32_t cb_gate_tiles = 2U * Sq_chunk_t * vWt;
    const uint32_t cb_intermediate_tiles = Sq_chunk_t * kIntermediateTiles;
    const uint32_t cb_attention_weights_tiles = Sq_chunk_t * Sk_chunk_t;  // single-buffered, rewritten in place
    const uint32_t cb_stat_tiles = Sq_chunk_t;                            // max / sum / exp-max-diff, one per row
    const uint32_t cb_mm_out_tiles = Sq_chunk_t * vWt;
    const uint32_t cb_output_tiles = Sq_chunk_t * vWt;

    [[maybe_unused]] auto cb_query = create_circular_buffer(
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_query_tiles);

    [[maybe_unused]] auto cb_key = create_circular_buffer(
        program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_key_tiles);

    [[maybe_unused]] auto cb_value = create_circular_buffer(
        program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_value_tiles);

    // Mask CB: causal = two on-chip tiles ([0] diagonal pattern, [1] all -1e9); arbitrary = Sq x Sk tiles
    // per step, double-buffered. None = no CB.
    if (mask_type != AttentionMaskType::None) {
        [[maybe_unused]] auto cb_attn_mask = create_circular_buffer(
            program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_attn_mask_tiles);
    }

    // Create gate buffer if the model has a gating mechanism. Double-buffered to overlap DRAM reads with compute.
    if (args.has_gate) {
        [[maybe_unused]] auto cb_gate = create_circular_buffer(
            program, all_cores, kGateCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_gate_tiles);
    }

    // create intermediate buffer only if we need to return intermediates
    // Intermediate shape: (B, H, S, 32) = 1 FP32 tile wide (logsumexp = max + log(sum_exp)) per row
    if (args.return_intermediates) {
        [[maybe_unused]] auto cb_intermediate = create_circular_buffer(
            program,
            all_cores,
            kIntermediateCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            cb_intermediate_tiles);
    }

    [[maybe_unused]] auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    // Scores of one step, row-major; single-buffered, softmax rewrites in place.
    //
    // TODO: once LLK provides SFPU row-reduce-max and sub_bcast_col, the softmax can run
    // entirely in DST at FP32 after the QK matmul, eliminating the pack/unpack round-trip
    // through this CB for the softmax critical path (~20 ms regression).
    [[maybe_unused]] auto cb_attention_weights = create_circular_buffer(
        program,
        all_cores,
        kAttentionWeightsCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        cb_attention_weights_tiles);

    [[maybe_unused]] auto cb_prev_max_value = create_circular_buffer(
        program, all_cores, kPrevMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_stat_tiles);

    [[maybe_unused]] auto cb_cur_max_value = create_circular_buffer(
        program, all_cores, kCurMaxValueCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_stat_tiles);

    // lets try to use precise data format for holding exp sum/diff values
    [[maybe_unused]] auto cb_exp_max_diff = create_circular_buffer(
        program, all_cores, kExpMaxDiffCbIndex, precise_data_format, float32_single_tile_size_bytes, cb_stat_tiles);

    [[maybe_unused]] auto cb_prev_exp_sum = create_circular_buffer(
        program, all_cores, kPrevSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, cb_stat_tiles);

    [[maybe_unused]] auto cb_cur_exp_sum = create_circular_buffer(
        program, all_cores, kCurSumExpCbIndex, precise_data_format, float32_single_tile_size_bytes, cb_stat_tiles);

    [[maybe_unused]] auto cb_prev_mm_out = create_circular_buffer(
        program, all_cores, kPrevMmOutCbIndex, precise_data_format, float32_single_tile_size_bytes, cb_mm_out_tiles);

    [[maybe_unused]] auto cb_cur_mm_out = create_circular_buffer(
        program, all_cores, kCurMmOutCbIndex, precise_data_format, float32_single_tile_size_bytes, cb_mm_out_tiles);

    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_output_tiles);

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

    auto* gate_buffer = args.has_gate ? gate.value().buffer() : nullptr;
    if (gate_buffer != nullptr) {
        TT_FATAL(
            gate_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Gate buffer must be in DRAM. Gate buffer of type {}",
            enchantum::to_string(gate_buffer->buffer_type()));
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

    // Set defines based on mask type (mutually exclusive)
    switch (mask_type) {
        case AttentionMaskType::Causal: defines[kCausalMaskDefKey] = "1"; break;
        case AttentionMaskType::Arbitrary: defines[kUseAttnMaskDefKey] = "1"; break;
        case AttentionMaskType::None:
            // No mask define - kernel will just scale attention scores
            break;
    }

    if (args.has_gate) {
        defines[kHasGateDefKey] = "1";
    }

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    SDPAForwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // num tile in inner dim in query/key (d_qk/TILE_W)
        vWt,              // num tile in inner dim in value (d_v/TILE_W)
        St,               // num tile in seq len dim (S/TILE_H)
        qNH,              // number of heads in query
        heads_per_group,  // number of heads per group
        Sk_chunk_t,       // K/V tiles per inner-loop step
        Sq_chunk_t,       // query tile-rows per Q chunk
    };
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ reader_compile_args,
        defines,
        kReaderKernelPath);

    std::vector<uint32_t> writer_compile_args = {
        vWt,         // num tile in inner dim in output/value (d_v/TILE_W)
        St,          // num tile in seq len dim (S/TILE_H)
        qNH,         // number of heads in query
        Sq_chunk_t,  // query tile-rows per Q chunk
    };
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(writer_compile_args);
    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ writer_compile_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // UnpackToDestFp32 for mm_out accum buffers and broadcast scratch:
    // copy_tile from these CBs bypasses SrcA and DMAs directly to DST at full FP32.
    // Only SFPU operations (mul_binary_tile, etc.) may be used on data loaded this way.
    auto create_unpack_to_dest_mode = []() {
        std::vector<tt::tt_metal::UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        mode[kPrevMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        mode[kCurMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        return mode;
    };

    const std::vector<uint32_t> compute_compile_args = {
        block_size,     // output tiles per DST pass in the element-wise loops
        qWt,            // num tile in inner dim in query/key (d_qk/TILE_W)
        vWt,            // num tile in inner dim in value (d_v/TILE_W)
        St,             // num_seq_len / TILE_H
        scaler,         // sqrt(Et) - sdpa scaler factor
        minus_one,      // used to transform mask from 1/0 to 0/-1
        custom_inf,     // used to transform mask from 0/-1 to 0/-1e9F
        Sk_chunk_t,     // K/V tiles per inner-loop step
        pv_block_w,     // V feature tiles per PV matmul_block call
        scaler_bf16,    // BF16 (RNE) scale for the WH fused scale+exp path
        Sq_chunk_t,     // query tile-rows per Q chunk
        qk_block_h,     // Q rows per QK^T matmul_block call
        qk_block_w,     // K tiles per QK^T matmul_block call
        pv_block_h,     // Q rows per PV matmul_block call
        sm_block_size,  // score tiles of one row per DST pass in the exp step
    };
    kernels.compute = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = create_unpack_to_dest_mode(),
            .math_approx_mode = false,
            .compile_args = compute_compile_args,
            .defines = defines});

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
        gate_buffer,
        output_buffer,
        intermediates_buffer,
        num_cores_y,
        per_core_runs);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* sdpa_fw_reader_kernel      = */ kernels.reader,
         /* sdpa_fw_writer_kernel      = */ kernels.writer,
         /* sdpa_fw_compute_kernel     = */ kernels.compute,
         /* num_cores                  = */ num_cores,
         /* num_cores_y                = */ num_cores_y}};
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

    const uint32_t num_cores = shared_vars.num_cores;
    const uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto* query_buffer = tensor_args.query.buffer();
    const auto* key_buffer = tensor_args.key.buffer();
    const auto* value_buffer = tensor_args.value.buffer();
    const auto* mask_buffer = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer() : nullptr;
    const auto* gate_buffer = tensor_args.gate.has_value() ? tensor_args.gate.value().buffer() : nullptr;
    auto* output_buffer = tensor_return_value.front().buffer();
    auto* intermediates_buffer =
        operation_attributes.return_intermediates ? tensor_return_value.back().buffer() : nullptr;

    // Only address arguments need updating here; tile counts remain the same as in create().
    // No runtime args to update for compute kernels.
    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_fw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_fw_writer_kernel);

    for_each_core(num_cores, num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kQueryBufferIdx] = query_buffer->address();
            runtime_args[kKeyBufferIdx] = key_buffer->address();
            runtime_args[kValueBufferIdx] = value_buffer->address();
            // For causal mask, mask is generated on-chip (mask_buffer is nullptr)
            // For DRAM mask, use actual mask buffer address
            runtime_args[kMaskBufferIdx] = (mask_buffer != nullptr) ? mask_buffer->address() : 0;
            runtime_args[kGateBufferIdx] = (gate_buffer != nullptr) ? gate_buffer->address() : 0;
        }

        // Update output buffer for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
            runtime_args[kIntermediateBufferIdx] =
                intermediates_buffer != nullptr ? intermediates_buffer->address() : 0;
        }
    });
}

}  // namespace ttml::metal::ops::sdpa_fw::device
