// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <utility>
#include <tt-metalium/bfloat16.hpp>
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
constexpr uint32_t kIntermediateTiles = 1U;  // logsumexp = max + log(sum_exp), single FP32 tile

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";
const std::string kBalancedParallelismDefKey = "BALANCED_PARALLELISM";

const std::string kHasGateDefKey = "HAS_GATE";

// Default K/V chunk when k_chunk_size is left at 0: the largest power of two in {1, 2, 4}
// that divides St. Four tiles is the fp32 DST budget, so this keeps the softmax exp pass to
// a single DST cycle; an explicit k_chunk_size may go past it and the kernel then splits the
// exp pass into sm_block_size-tile DST cycles.
uint32_t pick_default_sk_chunk_t(uint32_t St) {
    constexpr uint32_t kMaxSkChunkT = 4U;
    for (uint32_t c = kMaxSkChunkT; c > 1U; c /= 2U) {
        if (St % c == 0U) {
            return c;
        }
    }
    return 1U;
}

// Output tile rectangle (rt_dim x ct_dim) for one matmul_block call: rows must divide
// `rows`, cols must divide `cols`, and the product must fit the DST budget. Largest area
// first; between equal areas prefer more columns (contiguous in1 walk, fewer in0 re-unpacks).
std::pair<uint32_t, uint32_t> pick_matmul_block_dims(uint32_t rows, uint32_t cols, uint32_t dst_size) {
    std::pair<uint32_t, uint32_t> best{1U, 1U};
    for (uint32_t h = 1U; h <= rows; ++h) {
        if (rows % h != 0U) {
            continue;
        }
        for (uint32_t w = 1U; w <= cols; ++w) {
            if (cols % w != 0U || h * w > dst_size) {
                continue;
            }
            const uint32_t area = h * w;
            const uint32_t best_area = best.first * best.second;
            if (area > best_area || (area == best_area && w > best.second)) {
                best = {h, w};
            }
        }
    }
    return best;
}

/**
 * Calculate work-balanced pair distribution for causal SDPA.
 *
 * For causal attention, Q chunk i within a sequence attends to roughly (i+1) K/V chunks.
 * By pairing chunk i with chunk (q_num_chunks-1-i), each pair has near-constant work.
 * With one tile per Q chunk this is exactly the row pairing (St+1 K/V tiles per pair).
 *
 * @param total_pairs Total number of pairs = NC * q_num_chunks / 2
 * @param num_cores Number of cores to distribute across
 * @return Vector of (start_pair_idx, num_pairs) for each core
 */
std::vector<std::pair<uint32_t, uint32_t>> calculate_balanced_pair_distribution(
    uint32_t total_pairs, uint32_t num_cores) {
    TT_FATAL(num_cores > 0, "calculate_balanced_pair_distribution: num_cores must be > 0");
    std::vector<std::pair<uint32_t, uint32_t>> distribution;
    distribution.reserve(num_cores);

    const uint32_t pairs_per_core = total_pairs / num_cores;
    const uint32_t remainder = total_pairs % num_cores;

    uint32_t current_pair = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        // First 'remainder' cores get one extra pair
        const uint32_t pairs_for_this_core = pairs_per_core + (core < remainder ? 1 : 0);
        distribution.emplace_back(current_pair, pairs_for_this_core);
        current_pair += pairs_for_this_core;
    }

    return distribution;
}

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
 * Set up the runtime arguments for balanced parallelism mode.
 * In this mode, work is distributed by pairs of Q chunks rather than single Q chunks.
 * Each pair consists of a "light" chunk (early in sequence, less work) and
 * a "heavy" chunk (late in sequence, more work), providing balanced total work.
 */
void assign_per_core_runtime_args_balanced(
    tt::tt_metal::Program& program,
    const SDPAForwardKernels& kernels,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* gate_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const std::vector<std::pair<uint32_t, uint32_t>>& pair_distribution) {
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& [start_pair_idx, num_pairs] = pair_distribution[i];

        // Reader kernel: (query_addr, key_addr, value_addr, mask_addr(unused), gate_addr (optional), num_pairs,
        // start_pair_idx)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {query_buffer->address(),
             key_buffer->address(),
             value_buffer->address(),
             0,  // mask_addr unused for balanced causal - mask generated on chip
             gate_buffer != nullptr ? gate_buffer->address() : 0,
             num_pairs,
             start_pair_idx});

        // Writer kernel: (output_addr, intermediates_addr, num_pairs, start_pair_idx)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(),
             intermediates_buffer != nullptr ? intermediates_buffer->address() : 0,
             num_pairs,
             start_pair_idx});

        // Compute kernel: (start_pair_idx, num_pairs)
        SetRuntimeArgs(program, kernels.compute_group_1, core, {start_pair_idx, num_pairs});
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

    // -------------------------------------------------------------------------
    // Chunk geometry (the two program-config knobs, in tiles)
    // -------------------------------------------------------------------------
    // Sq_chunk_t query tile-rows share one pass over K/V: the reader fetches each K/V chunk
    // once per Q chunk instead of once per row. Sk_chunk_t K/V tiles are consumed per
    // inner-loop step. The device operation already checked that explicit sizes are tile
    // multiples dividing the sequence; 0 keeps the historical defaults (1 row, up to 4 K tiles).
    const uint32_t Sq_chunk_t = args.q_chunk_size == 0U ? 1U : args.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t =
        args.k_chunk_size == 0U ? pick_default_sk_chunk_t(St) : args.k_chunk_size / tt::constants::TILE_HEIGHT;
    TT_FATAL(Sq_chunk_t > 0U && (St % Sq_chunk_t == 0U), "Sq_chunk_t={} must be > 0 and divide St={}", Sq_chunk_t, St);
    TT_FATAL(Sk_chunk_t > 0U && (St % Sk_chunk_t == 0U), "Sk_chunk_t={} must be > 0 and divide St={}", Sk_chunk_t, St);

    // The unit of work handed to a core is one Q chunk (Sq_chunk_t rows of one batch/head).
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

    // -------------------------------------------------------------------------
    // Balanced Parallelism Decision
    // -------------------------------------------------------------------------
    // For causal SDPA, work per Q chunk grows with its position in the sequence: chunk i
    // attends to about (i+1) K/V chunks. Pairing chunk i with chunk (q_num_chunks-1-i)
    // (first with last, second with second-to-last, ...) makes every pair cost about the
    // same, which is what balances the cores.
    //
    // Conditions for balanced parallelism:
    // 1. Causal mask (triangular workload pattern)
    // 2. Even number of Q chunks per sequence (q_num_chunks % 2 == 0) for clean pairing
    // 3. Enough work for all cores (total_pairs >= num_available_cores)
    const uint32_t pairs_per_seq = q_num_chunks / 2;
    const uint32_t total_pairs = NC * pairs_per_seq;
    const bool use_balanced_parallelism = (mask_type == AttentionMaskType::Causal) && (q_num_chunks % 2 == 0) &&
                                          (total_pairs >= num_available_cores) && (St > 0);

    // Variables for work distribution (will be set based on parallelism mode)
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_units_per_core_group_1 = 0;
    uint32_t num_units_per_core_group_2 = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pair_distribution;

    if (use_balanced_parallelism) {
        // Use all available cores for balanced distribution
        num_cores = std::min(num_available_cores, total_pairs);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
        core_group_1 = all_cores;  // All cores in one group for balanced mode

        // Calculate pair distribution across cores
        pair_distribution = calculate_balanced_pair_distribution(total_pairs, num_cores);
    } else {
        // Standard distribution of Q chunks over cores
        auto work_split = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_q_chunks);
        num_cores = std::get<0>(work_split);
        all_cores = std::get<1>(work_split);
        core_group_1 = std::get<2>(work_split);
        core_group_2 = std::get<3>(work_split);
        num_units_per_core_group_1 = std::get<4>(work_split);
        num_units_per_core_group_2 = std::get<5>(work_split);
    }

    // DST register budget. fp32_dest_acc_en=true → 4 slots (FP32 accumulator);
    // fp32_dest_acc_en=false → 8 slots (BF16). Mirrors TTNN's `dst_size` convention in their
    // SDPA program factory. Derived granularities (all "block" = tiles held in DST per cycle):
    //
    //   * `block_size`   (cap = dst_size - 1) — SFPU ops on one row's output tiles
    //     (`update_cur_mm_out`, final normalization): `block_size` data tiles plus a 1-tile
    //     scratch for `unary_bcast<COL>`. Divides vWt.
    //   * `qk_block_h x qk_block_w` — one QK^T `matmul_block` call: Q rows x K tiles,
    //     dividing Sq_chunk_t x Sk_chunk_t, product <= dst_size.
    //   * `pv_block_h x pv_block_w` — one PV `matmul_block` call: Q rows x V feature tiles,
    //     dividing Sq_chunk_t x vWt, product <= dst_size.
    //   * `sm_block_size` (cap = dst_size) — score tiles of one row per DST cycle of the
    //     softmax sub+exp pass. Divides Sk_chunk_t.
    constexpr bool fp32_dest_acc_en = true;
    const uint32_t dst_size = fp32_dest_acc_en ? 4U : 8U;
    const uint32_t block_size = get_block_size(vWt, dst_size - 1U);
    const auto qk_block_dims = pick_matmul_block_dims(Sq_chunk_t, Sk_chunk_t, dst_size);
    const uint32_t qk_block_h = qk_block_dims.first;
    const uint32_t qk_block_w = qk_block_dims.second;
    const auto pv_block_dims = pick_matmul_block_dims(Sq_chunk_t, vWt, dst_size);
    const uint32_t pv_block_h = pv_block_dims.first;
    const uint32_t pv_block_w = pv_block_dims.second;
    const uint32_t sm_block_size = get_block_size(Sk_chunk_t, dst_size);

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    // Tile counts per CB. Q, K, V, gate and the arbitrary mask are double-buffered so the
    // reader overlaps the next chunk's DRAM reads with compute on the current one. Everything
    // the softmax keeps per query row is Sq_chunk_t deep: row r of the chunk owns slot r.
    const uint32_t cb_query_tiles = 2U * Sq_chunk_t * qWt;
    const uint32_t cb_key_tiles = 2U * Sk_chunk_t * kWt;
    const uint32_t cb_value_tiles = 2U * Sk_chunk_t * vWt;
    const uint32_t cb_attn_mask_tiles =
        (mask_type == AttentionMaskType::Causal) ? kNumCausalMaskTiles : (2U * Sq_chunk_t * Sk_chunk_t);
    const uint32_t cb_gate_tiles = 2U * Sq_chunk_t * vWt;
    const uint32_t cb_intermediate_tiles = Sq_chunk_t * kIntermediateTiles;
    const uint32_t cb_attention_weights_tiles = Sq_chunk_t * Sk_chunk_t;  // single-buffered, rewritten in place
    const uint32_t cb_stat_tiles = Sq_chunk_t;                           // max / sum / exp-max-diff, one per row
    const uint32_t cb_mm_out_tiles = Sq_chunk_t * vWt;
    const uint32_t cb_output_tiles = Sq_chunk_t * vWt;

    {
        // Every CB lives in L1 at once; refuse configurations that do not fit instead of
        // letting the allocator fail inside program creation with a less useful message.
        const uint64_t bf16 = bfloat16_single_tile_size_bytes;
        const uint64_t fp32 = float32_single_tile_size_bytes;
        uint64_t l1_bytes = bf16 * (cb_query_tiles + cb_key_tiles + cb_value_tiles + 2U * kNumScalerTiles +
                                    2U * cb_stat_tiles + cb_output_tiles) +
                            fp32 * (cb_attention_weights_tiles + 3U * cb_stat_tiles + 2U * cb_mm_out_tiles);
        if (mask_type != AttentionMaskType::None) {
            l1_bytes += bf16 * cb_attn_mask_tiles;
        }
        if (args.has_gate) {
            l1_bytes += bf16 * cb_gate_tiles;
        }
        if (args.return_intermediates) {
            l1_bytes += fp32 * cb_intermediate_tiles;
        }
        const uint64_t l1_budget_bytes =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        TT_FATAL(
            l1_bytes <= l1_budget_bytes,
            "SDPA forward circular buffers need {} bytes of L1 per core but only {} are available. "
            "Reduce q_chunk_size ({} rows) or k_chunk_size ({} rows).",
            l1_bytes,
            l1_budget_bytes,
            Sq_chunk_t * tt::constants::TILE_HEIGHT,
            Sk_chunk_t * tt::constants::TILE_HEIGHT);
    }

    [[maybe_unused]] auto cb_query = create_circular_buffer(
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_query_tiles);

    [[maybe_unused]] auto cb_key = create_circular_buffer(
        program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_key_tiles);

    [[maybe_unused]] auto cb_value = create_circular_buffer(
        program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, cb_value_tiles);

    // create mask buffer if using attention mask from DRAM or generating causal mask on-the-fly.
    // Not needed for AttentionMaskType::None.
    //
    // Sizing depends on the mask type:
    //   - Causal (incl. balanced parallelism): two persistently-fronted pre-transformed tiles
    //     generated on chip
    //       [0] = causal-diagonal (0.0 on/below diagonal, -1e9 above) — applied where the
    //             K tile index equals the query row's tile index.
    //       [1] = all -1e9 — applied on K tiles strictly past a row's diagonal inside the
    //             chunks that overlap the diagonal band of the Q chunk.
    //   - Arbitrary: Sq_chunk_t x Sk_chunk_t mask tiles per step are streamed from DRAM;
    //     double-buffer them.
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

    // Holds the Sq_chunk_t x Sk_chunk_t QK^T score tiles for the current step (row-major:
    // row r, K tile n at r * Sk_chunk_t + n). Sized to exactly one step (not double-buffered)
    // since softmax overwrites in-place, then the PV matmul consumes it.
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

    // Add balanced parallelism define if using that mode
    if (use_balanced_parallelism) {
        defines[kBalancedParallelismDefKey] = "1";
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
        vWt,  // num tile in inner dim in output/value (d_v/TILE_W)
        St,   // num tile in seq len dim (S/TILE_H)
        qNH,  // number of heads in query
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

    // Compile-time args shared by every compute kernel instance; only the per-core unit count
    // (slot 0: Q chunks in standard mode, pairs in balanced mode) differs between core groups.
    const auto make_compute_args = [&](uint32_t units_per_core) {
        return std::vector<uint32_t>{
            units_per_core,  // per_core unit count (Q chunks, or pairs in balanced mode)
            block_size,      // SFPU per-row block size (cap = dst_size - 1, needs scratch)
            qWt,             // num tile in inner dim in query/key (d_qk/TILE_W)
            vWt,             // num tile in inner dim in value (d_v/TILE_W)
            St,              // num_seq_len / TILE_H
            scaler,          // sqrt(Et) - sdpa scaler factor
            minus_one,       // used to transform mask from 1/0 to 0/-1
            custom_inf,      // used to transform mask from 0/-1 to 0/-1e9F
            Sk_chunk_t,      // K/V tiles per inner-loop step
            pv_block_w,      // PV matmul_block ct_dim (V feature tiles per call)
            scaler_bf16,     // BF16 (RNE) scale for the WH fused scale+exp path
            Sq_chunk_t,      // query tile-rows per Q chunk
            qk_block_h,      // QK^T matmul_block rt_dim (Q rows per call)
            qk_block_w,      // QK^T matmul_block ct_dim (K tiles per call)
            pv_block_h,      // PV matmul_block rt_dim (Q rows per call)
            sm_block_size,   // score tiles per DST cycle in the softmax exp pass
        };
    };
    const auto make_compute_config = [&](std::vector<uint32_t> compile_args) {
        return tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = create_unpack_to_dest_mode(),
            .math_approx_mode = false,
            .compile_args = std::move(compile_args),
            .defines = defines};
    };

    if (use_balanced_parallelism) {
        // For balanced parallelism, all cores are in one group with uniform work.
        // num_pairs is passed as compile-time arg (max pairs per core); the runtime arg
        // carries each core's actual count.
        const uint32_t max_pairs_per_core = pair_distribution.empty() ? 0 : pair_distribution[0].second;
        kernels.compute_group_1 = tt::tt_metal::CreateKernel(
            program, kComputeKernelPath, all_cores, make_compute_config(make_compute_args(max_pairs_per_core)));
    } else {
        // Standard mode: two potential core groups with different Q chunk counts
        kernels.compute_group_1 = tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            core_group_1,
            make_compute_config(make_compute_args(num_units_per_core_group_1)));

        if (!core_group_2.ranges().empty()) {
            kernels.compute_group_2 = tt::tt_metal::CreateKernel(
                program,
                kComputeKernelPath,
                core_group_2,
                make_compute_config(make_compute_args(num_units_per_core_group_2)));
        }
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    if (use_balanced_parallelism) {
        assign_per_core_runtime_args_balanced(
            program,
            kernels,
            query_buffer,
            key_buffer,
            value_buffer,
            gate_buffer,
            output_buffer,
            intermediates_buffer,
            num_cores,
            num_cores_y,
            pair_distribution);
    } else {
        for_each_core_with_work(
            num_cores,
            num_cores_y,
            core_group_1,
            core_group_2,
            num_units_per_core_group_1,
            num_units_per_core_group_2,
            [&](const CoreWork& work) {
                const auto& [core, core_index, num_q_chunks, start_q_chunk, in_group_1] = work;
                SetRuntimeArgs(
                    program,
                    kernels.reader,
                    core,
                    {query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     mask_buffer != nullptr ? mask_buffer->address() : 0,
                     gate_buffer != nullptr ? gate_buffer->address() : 0,
                     num_q_chunks,
                     start_q_chunk});
                SetRuntimeArgs(
                    program,
                    kernels.writer,
                    core,
                    {output_buffer->address(),
                     intermediates_buffer != nullptr ? intermediates_buffer->address() : 0,
                     num_q_chunks,
                     start_q_chunk});
                SetRuntimeArgs(
                    program, in_group_1 ? kernels.compute_group_1 : kernels.compute_group_2, core, {start_q_chunk});
            });
    }

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* sdpa_fw_reader_kernel      = */ kernels.reader,
         /* sdpa_fw_writer_kernel      = */ kernels.writer,
         /* sdpa_fw_kernel_group_1     = */ kernels.compute_group_1,
         /* sdpa_fw_kernel_group_2     = */ kernels.compute_group_2,
         /* core_group_1               = */ core_group_1,
         /* core_group_2               = */ core_group_2,
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
