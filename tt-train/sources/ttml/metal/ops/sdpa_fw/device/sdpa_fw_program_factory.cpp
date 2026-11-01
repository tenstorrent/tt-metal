// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/kernel_types.hpp>
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
constexpr uint32_t kMaxValueHolderTiles = 1U;
constexpr uint32_t kExpMaxDiffTiles = 1U;
constexpr uint32_t kExpSumTiles = 1U;
constexpr uint32_t kIntermediateTiles = 1U;  // logsumexp = max + log(sum_exp), single FP32 tile

const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";
const std::string kBalancedParallelismDefKey = "BALANCED_PARALLELISM";
const std::string kWorkListDefKey = "WORK_LIST";
const std::string kChainDefKey = "CHAIN";
constexpr uint32_t kChainSlots = 3U;  // stream chunks resident per core (credits run kChainSlots - 1 ahead)
// Cores per K/V forwarding chain: 8 measured best at the TP=4 shard shape (iteration 26: 114 us against 131 with 14,
// 147 with 4 where the injectors' DRAM reads dominate); tunable for sweeps.
const uint32_t kChainLength = env_uint32_or("TTML_SDPA_FW_CHAIN_LENGTH", 8U);
constexpr uint32_t kNoRow = 0xFFFFFFFFU;  // "no row this round" in a chain-mode row list

const std::string kHasGateDefKey = "HAS_GATE";

// Pick the multi-tile K/V chunk size. Returns the number of K tiles processed per inner-loop
// iteration of the compute kernel.
//
// Constraints:
//   1. Must divide Ht (so the diagonal of the causal mask falls cleanly within a chunk,
//      and so the reader never reads past the end of the sequence).
//   2. The QK^T and exp stages process the chunk in DEST-sized passes (iteration 26), so the
//      chunk may exceed the DEST tile count; the PV matmul contracts the whole chunk in DEST.
//   3. Larger chunks amortize the softmax correction (and, in chain mode, the hop count) more,
//      but consume L1 proportionally for cb_key, cb_value, and cb_attention_weights.
//
// We pick the largest power-of-2 in {1, 2, 4, 8} that divides Ht.
uint32_t pick_sk_chunk_t(uint32_t Ht) {
    constexpr uint32_t kMaxSkChunkT = 8U;
    for (uint32_t c = kMaxSkChunkT; c > 1U; c /= 2U) {
        if (Ht % c == 0U) {
            return c;
        }
    }
    return 1U;
}

/**
 * Calculate work-balanced pair distribution for causal SDPA.
 *
 * For causal attention, row i within a sequence processes (i+1) K/V tiles.
 * By pairing row i with row (St-1-i), each pair has constant work = St+1.
 * This enables perfect work balance across cores.
 *
 * @param total_pairs Total number of pairs = NC * St / 2
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
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid. (Standard row-based distribution)
 */
/**
 * Set up the runtime arguments for balanced parallelism mode.
 * In this mode, work is distributed by pairs rather than rows.
 * Each pair consists of a "light" row (early in sequence, less work) and
 * a "heavy" row (late in sequence, more work), providing balanced total work.
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
    const uint32_t total_rows_to_process =
        NC * St;  // total rows to process = batch_size * num_heads * num_tiles_in_seq_len

    const uint32_t heads_per_group = qNH / kNH;  // we read heads_per_group heads from Q for one group of K and V
    const uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in inner dim
    const uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    const uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;

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
    // For causal SDPA, work per row follows triangular pattern: row i processes (i+1) K/V tiles.
    // By pairing row i with row (St-1-i), each pair has constant work = St+1.
    // This enables near-perfect work balance across cores.
    //
    // Conditions for balanced parallelism:
    // 1. Causal mask (triangular workload pattern)
    // 2. Even number of sequence tiles (St % 2 == 0) for clean pairing
    // 3. Enough work for all cores (total_pairs >= num_available_cores)
    const uint32_t pairs_per_seq = St / 2;
    const uint32_t total_pairs = NC * pairs_per_seq;
    // Work-list mode (causal): rows are scheduled longest-first onto the least loaded core (host LPT) and
    // each core receives its row list as runtime args. The pair scheme below balanced pairs, not cores: with
    // 128 pairs on 110 cores 18 cores did twice the work of the rest. Kept for reference; not selected.
    // Chain mode (causal): the cores are split per (batch, K/V group) stream; inside a stream the rows of its
    // query heads are LPT-scheduled over the stream's cores and the K/V chunks are read from DRAM once per chain
    // of kChainLength cores by its first core and forwarded core to core (variable_matmul's in0 protocol), so no
    // core re-reads K/V per query row. Every core runs the same number of rounds (one row or none per round).
    const bool causal = (mask_type == AttentionMaskType::Causal) && (St > 0);
    const uint32_t num_streams = kB * kNH;
    const bool use_chain = causal && (num_available_cores >= 2U * num_streams) && (St % pick_sk_chunk_t(St) == 0U);
    const bool use_work_list = causal && !use_chain;
    const bool use_balanced_parallelism = !causal && false;

    // Variables for work distribution (will be set based on parallelism mode)
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pair_distribution;

    if (use_work_list || use_chain) {
        num_cores = std::min(num_available_cores, total_rows_to_process);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, false);
        core_group_1 = all_cores;
    } else if (use_balanced_parallelism) {
        // Use all available cores for balanced distribution
        num_cores = std::min(num_available_cores, total_pairs);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
        core_group_1 = all_cores;  // All cores in one group for balanced mode

        // Calculate pair distribution across cores
        pair_distribution = calculate_balanced_pair_distribution(total_pairs, num_cores);

        // For balanced mode, num_rows_per_core is 2 * num_pairs (each pair = 2 rows)
        // But kernels will loop over pairs, not rows directly
        num_rows_per_core_group_1 = pair_distribution.empty() ? 0 : pair_distribution[0].second * 2;
    } else {
        // Standard row-based distribution (existing approach)
        auto work_split = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
        num_cores = std::get<0>(work_split);
        all_cores = std::get<1>(work_split);
        core_group_1 = std::get<2>(work_split);
        core_group_2 = std::get<3>(work_split);
        num_rows_per_core_group_1 = std::get<4>(work_split);
        num_rows_per_core_group_2 = std::get<5>(work_split);
    }

    // DST register budget. fp32_dest_acc_en=true → 4 slots (FP32 accumulator);
    // fp32_dest_acc_en=false → 8 slots (BF16). Mirrors TTNN's `dst_size` convention in their
    // SDPA program factory. Two derived granularities follow:
    //
    //   * `block_size`     (cap = dst_size - 1) — for SFPU ops (`update_cur_mm_out`,
    //     final normalization) that load `block_size` data tiles plus a 1-tile scratch
    //     for `unary_bcast<COL>`. Needs `block_size + 1 <= dst_size`.
    //   * `pv_block_size`  (cap = dst_size)     — for `matmul_block` in PV. The MOP writes
    //     `ct_dim` output tiles directly, no scratch needed; can use the full DST budget.
    //
    // Both are clamped to divisors of `vWt` by `get_block_size`, so the matmul outer loop
    // and the SFPU outer loop each tile `vWt` evenly without remainder handling.
    constexpr bool fp32_dest_acc_en = true;
    const uint32_t dst_size = fp32_dest_acc_en ? 4U : 8U;
    const uint32_t block_size = get_block_size(vWt, dst_size - 1U);
    const uint32_t pv_block_size = get_block_size(vWt, dst_size);

    // Multi-tile K/V chunking factor. Compute kernel processes Sk_chunk_t K and V tiles per
    // inner-loop iteration; reader pre-stages chunk-sized K/V blocks in L1.
    const uint32_t Sk_chunk_t = pick_sk_chunk_t(St);
    TT_FATAL(Sk_chunk_t > 0U && (St % Sk_chunk_t == 0U), "Sk_chunk_t={} must be > 0 and divide St={}", Sk_chunk_t, St);

    // Work-list schedule: row r of a sequence processes round_up(r + 1, Sk_chunk_t) K/V tiles plus a constant
    // (its Q read, normalisation and output write), weighted here as one tile.
    LptSchedule work_list;
    if (use_work_list) {
        std::vector<uint32_t> weights(total_rows_to_process);
        for (uint32_t row = 0; row < total_rows_to_process; ++row) {
            const uint32_t q_row_tile = row % St;
            weights[row] = ((q_row_tile + 1U + Sk_chunk_t - 1U) / Sk_chunk_t) * Sk_chunk_t + 1U;
        }
        work_list = lpt_schedule(weights, num_cores);
        TT_FATAL(
            work_list.max_items + 16U <= tt::tt_metal::max_runtime_args,
            "SDPA work list: a core would receive {} items, more than the runtime-arg limit allows",
            work_list.max_items);
        num_rows_per_core_group_1 = work_list.max_items;
    }

    // Chain schedule: per stream (batch, K/V group) a contiguous run of cores; LPT of the stream's rows over them;
    // chains of kChainLength consecutive cores, the first core of a chain the injector, the last the sink.
    struct ChainCore {
        uint32_t stream = 0;
        bool injector = false;
        bool sink = false;
        uint32_t prev = 0;   // linear index of the core it receives from
        uint32_t next = 0;   // linear index of the core it forwards to
        uint32_t chain = 0;  // global chain index
        std::vector<uint32_t> rows;
        std::vector<uint32_t> stream_chunks;  // per round: the chunks the chain streams (iteration 27)
    };
    std::vector<ChainCore> chain_cores;
    uint32_t chain_rounds = 0;
    uint32_t num_chains = 0;
    if (use_chain) {
        chain_cores.resize(num_cores);
        const uint32_t rows_per_stream = total_rows_to_process / num_streams;
        const uint32_t base_cores = num_cores / num_streams;
        const uint32_t extra_cores = num_cores % num_streams;
        uint32_t first_core = 0;
        for (uint32_t stream = 0; stream < num_streams; ++stream) {
            const uint32_t stream_cores = base_cores + (stream < extra_cores ? 1U : 0U);
            // Rows of the stream: query heads [kv * heads_per_group, (kv + 1) * heads_per_group) of batch b.
            const uint32_t b = stream / kNH;
            const uint32_t kv = stream % kNH;
            std::vector<uint32_t> rows;
            std::vector<uint32_t> weights;
            rows.reserve(rows_per_stream);
            for (uint32_t h = kv * heads_per_group; h < (kv + 1U) * heads_per_group; ++h) {
                for (uint32_t q = 0; q < St; ++q) {
                    rows.push_back((b * qNH + h) * St + q);
                    weights.push_back(((q + 1U + Sk_chunk_t - 1U) / Sk_chunk_t) * Sk_chunk_t + 1U);
                }
            }
            const auto schedule = lpt_schedule(weights, stream_cores);
            chain_rounds = std::max(chain_rounds, schedule.max_items);
            for (uint32_t c = 0; c < stream_cores; ++c) {
                auto& cc = chain_cores[first_core + c];
                cc.stream = stream;
                cc.chain = num_chains + c / kChainLength;
                const uint32_t pos = c % kChainLength;
                // The chain runs downward in core index (iteration 23): the writer forwards on NOC_1, whose
                // routes decrease in coordinate, so the next core is the previous index (one link).
                cc.injector = (pos == kChainLength - 1U) || (c + 1U == stream_cores);
                cc.sink = (pos == 0U);
                cc.prev = cc.injector ? first_core + c : first_core + c + 1U;
                cc.next = cc.sink ? first_core + c : first_core + c - 1U;
                for (const uint32_t item : schedule.items_per_core[c]) {
                    cc.rows.push_back(rows[item]);
                }
            }
            first_core += stream_cores;
            num_chains += (stream_cores + kChainLength - 1U) / kChainLength;
        }
        // Per chain and round the stream stops at the diagonal chunk of the round's heaviest row (iteration 27):
        // every core of the chain consumes the same chunks, so the chain's rows decide.
        std::vector<std::vector<uint32_t>> chain_chunks(num_chains, std::vector<uint32_t>(chain_rounds, 0U));
        for (const auto& cc : chain_cores) {
            for (uint32_t r = 0; r < cc.rows.size(); ++r) {
                const uint32_t q = cc.rows[r] % St;
                chain_chunks[cc.chain][r] =
                    std::max(chain_chunks[cc.chain][r], (q + 1U + Sk_chunk_t - 1U) / Sk_chunk_t);
            }
        }
        for (auto& cc : chain_cores) {
            cc.stream_chunks = chain_chunks[cc.chain];
        }
        TT_FATAL(
            2U * chain_rounds + 24U <= tt::tt_metal::max_runtime_args,
            "SDPA chain: {} rounds exceed the runtime-arg limit",
            chain_rounds);
        num_rows_per_core_group_1 = chain_rounds;
    }
    const uint32_t chain_sender_sem = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    const uint32_t chain_receiver_sem = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // The VALID relay of iterations 15-25 was replaced by the counting receiver semaphore (iteration 26); its
    // compile-arg slot stays at 0 so the argument indices after it are unchanged.
    constexpr uint32_t chain_valid_sem = 0U;
    const uint32_t chain_received_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);   // reader -> writer
    const uint32_t chain_forwarded_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);  // writer -> reader

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_query = create_circular_buffer(
        program, all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    // K and V are read in Sk_chunk_t-tile chunks. Double-buffer the chunks so the reader
    // can overlap next-chunk DRAM with current-chunk compute.
    [[maybe_unused]] auto cb_key = create_circular_buffer(
        program,
        all_cores,
        kKeyCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        (use_chain ? kChainSlots : 2U) * Sk_chunk_t * kWt);

    [[maybe_unused]] auto cb_value = create_circular_buffer(
        program,
        all_cores,
        kValueCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        (use_chain ? kChainSlots : 2U) * Sk_chunk_t * vWt);

    // create mask buffer if using attention mask from DRAM or generating causal mask on-the-fly.
    // Not needed for AttentionMaskType::None.
    //
    // Sizing depends on the mask type:
    //   - Causal (incl. balanced parallelism): two persistently-fronted pre-transformed tiles
    //     generated on chip
    //       [0] = causal-diagonal (0.0 on/below diagonal, -1e9 above) — applied on the
    //             diagonal K tile of every row.
    //       [1] = all -1e9 — applied on K tiles strictly past the diagonal inside the
    //             diagonal chunk when Sk_chunk_t > 1.
    //   - Arbitrary: Sk_chunk_t mask tiles per K chunk are streamed from DRAM; double-buffer them.
    if (mask_type != AttentionMaskType::None) {
        const uint32_t num_attn_mask_tiles =
            (mask_type == AttentionMaskType::Causal) ? kNumCausalMaskTiles : (2U * Sk_chunk_t);
        [[maybe_unused]] auto cb_attn_mask = create_circular_buffer(
            program, all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, num_attn_mask_tiles);
    }

    // Create gate buffer if the model has a gating mechanism. Double-buffered to overlap DRAM reads with compute.
    if (args.has_gate) {
        [[maybe_unused]] auto cb_gate = create_circular_buffer(
            program, all_cores, kGateCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);
    }

    // create intermediate buffer only if we need to return intermediates
    // Intermediate shape: (B, H, S, 32) = 1 FP32 tile wide (logsumexp = max + log(sum_exp))
    if (args.return_intermediates) {
        [[maybe_unused]] auto cb_intermediate = create_circular_buffer(
            program,
            all_cores,
            kIntermediateCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kIntermediateTiles);
    }

    [[maybe_unused]] auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    [[maybe_unused]] auto cb_mat_mul_reduce = create_circular_buffer(
        program, all_cores, kMatMulReduceCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    // Holds the Sk_chunk_t QK^T score tiles of the current K chunk and, in the second slot, their exp
    // (iteration 26: the exp stage writes the free slot and pops the raw scores; the PV matmul consumes it).
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
        2U * Sk_chunk_t);

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
        program, all_cores, kPrevMmOutCbIndex, precise_data_format, float32_single_tile_size_bytes, vWt);

    [[maybe_unused]] auto cb_cur_mm_out = create_circular_buffer(
        program, all_cores, kCurMmOutCbIndex, precise_data_format, float32_single_tile_size_bytes, vWt);

    [[maybe_unused]] auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, vWt);

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
    if (use_work_list || use_chain) {
        defines[kWorkListDefKey] = "1";  // the row-list decoding of the compute and writer kernels
    }
    if (use_chain) {
        defines[kChainDefKey] = "1";
        defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);
    }

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    SDPAForwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,                  // num tile in inner dim in query/key (d_qk/TILE_W)
        vWt,                  // num tile in inner dim in value (d_v/TILE_W)
        St,                   // num tile in seq len dim (S/TILE_H)
        qNH,                  // number of heads in query
        heads_per_group,      // number of heads per group
        Sk_chunk_t,           // multi-tile K/V chunking factor
        chain_sender_sem,     // chain mode: "next core is ready" counter (incremented by the next core)
        chain_receiver_sem,   // chain mode: "data valid" flag (set by the previous core)
        chain_valid_sem,      // unused slot (the VALID relay, replaced in iteration 26)
        chain_received_sem,   // chain mode: chunks the reader has received (to the writer)
        chain_forwarded_sem,  // chain mode: chunks the writer has forwarded (to the reader)
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
        vWt,               // num tile in inner dim in output/value (d_v/TILE_W)
        St,                // num tile in seq len dim (S/TILE_H)
        qNH,               // number of heads in query
        Sk_chunk_t,        // K/V chunk tiles (the writer forwards the chain's chunks)
        qWt,               // num tile in inner dim in query/key
        chain_sender_sem,  // chain mode semaphores
        chain_receiver_sem,
        chain_valid_sem,
        chain_received_sem,
        chain_forwarded_sem,
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

    if (use_balanced_parallelism) {
        // For balanced parallelism, all cores are in one group with uniform work
        // num_pairs is passed as compile-time arg (max pairs per core)
        const uint32_t max_pairs_per_core = pair_distribution.empty() ? 0 : pair_distribution[0].second;

        std::vector<uint32_t> compute_args = {
            max_pairs_per_core,  // num_pairs (or max_pairs if some cores have fewer)
            block_size,          // SFPU per-core block size (cap = dst_size - 1, needs scratch)
            qWt,                 // num tile in inner dim in query/key (d_qk/TILE_W)
            vWt,                 // num tile in inner dim in value (d_v/TILE_W)
            St,                  // num_seq_len / TILE_H
            scaler,              // sqrt(Et) - sdpa scaler factor
            minus_one,           // used to transform mask from 1/0 to 0/-1
            custom_inf,          // used to transform mask from 0/-1 to 0/-1e9F
            Sk_chunk_t,          // multi-tile K/V chunking factor
            pv_block_size,       // PV matmul_block ct_dim (cap = dst_size, no scratch)
            scaler_bf16,         // BF16 (RNE) scale for the WH fused scale+exp path
            dst_size,            // DEST tiles per pass (QK^T / exp passes of the chunk)
        };

        kernels.compute_group_1 = tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = create_unpack_to_dest_mode(),
                .math_approx_mode = false,
                .compile_args = compute_args,
                .defines = defines});
    } else {
        // Standard mode: two potential core groups with different row counts

        // Group 1 compile-time arguments
        std::vector<uint32_t> compute_group_1_args = {
            num_rows_per_core_group_1,  // per_core_block_cnt
            block_size,                 // SFPU per-core block size (cap = dst_size - 1, needs scratch)
            qWt,                        // num tile in inner dim in query/key (d_qk/TILE_W)
            vWt,                        // num tile in inner dim in value (d_v/TILE_W)
            St,                         // num_seq_len / TILE_H
            scaler,                     // sqrt(Et) - sdpa scaler factor
            minus_one,                  // used to transform mask from 1/0 to 0/-1
            custom_inf,                 // used to transform mask from 0/-1 to 0/-1e9F
            Sk_chunk_t,                 // multi-tile K/V chunking factor
            pv_block_size,              // PV matmul_block ct_dim (cap = dst_size, no scratch)
            scaler_bf16,                // BF16 (RNE) scale for the WH fused scale+exp path
            dst_size,                   // DEST tiles per pass (QK^T / exp passes of the chunk)
        };

        kernels.compute_group_1 = tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            core_group_1,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = create_unpack_to_dest_mode(),
                .math_approx_mode = false,
                .compile_args = compute_group_1_args,
                .defines = defines});

        // Group 2 (if present) compile-time arguments
        if (!core_group_2.ranges().empty()) {
            std::vector<uint32_t> compute_group_2_args = {
                num_rows_per_core_group_2,  // per_core_block_cnt
                block_size,                 // SFPU per-core block size (cap = dst_size - 1, needs scratch)
                qWt,                        // num tile in inner dim in query/key (d_qk/TILE_W)
                vWt,                        // num tile in inner dim in value (d_v/TILE_W)
                St,                         // num_seq_len / TILE_H
                scaler,                     // sqrt(Et) - sdpa scaler factor
                minus_one,                  // used to transform mask from 1/0 to 0/-1
                custom_inf,                 // used to transform mask from 0/-1 to 0/-1e9F
                Sk_chunk_t,                 // multi-tile K/V chunking factor
                pv_block_size,              // PV matmul_block ct_dim (cap = dst_size, no scratch)
                scaler_bf16,                // BF16 (RNE) scale for the WH fused scale+exp path
                dst_size,                   // DEST tiles per pass (QK^T / exp passes of the chunk)
            };

            kernels.compute_group_2 = tt::tt_metal::CreateKernel(
                program,
                kComputeKernelPath,
                core_group_2,
                tt::tt_metal::ComputeConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = true,
                    .unpack_to_dest_mode = create_unpack_to_dest_mode(),
                    .math_approx_mode = false,
                    .compile_args = compute_group_2_args,
                    .defines = defines});
        }
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    if (use_chain) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto& cc = chain_cores[i];
            std::vector<uint32_t> rounds(cc.rows);
            rounds.resize(chain_rounds, kNoRow);
            // The rows (padded) then, per round, the chunks the chain streams.
            const auto with_chunks = [&](std::vector<uint32_t> args) {
                args.insert(args.end(), cc.stream_chunks.begin(), cc.stream_chunks.end());
                return args;
            };
            const auto prev = device->worker_core_from_logical_core(
                tt::tt_metal::CoreCoord{cc.prev / num_cores_y, cc.prev % num_cores_y});
            const auto next = device->worker_core_from_logical_core(
                tt::tt_metal::CoreCoord{cc.next / num_cores_y, cc.next % num_cores_y});
            SetRuntimeArgs(
                program,
                kernels.reader,
                core,
                with_chunks(work_list_runtime_args(
                    {query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     0,  // mask generated on chip
                     gate_buffer != nullptr ? gate_buffer->address() : 0,
                     static_cast<uint32_t>(prev.x),
                     static_cast<uint32_t>(prev.y),
                     static_cast<uint32_t>(next.x),
                     static_cast<uint32_t>(next.y),
                     cc.injector ? 1U : 0U,
                     cc.sink ? 1U : 0U,
                     cc.stream},
                    rounds)));
            SetRuntimeArgs(
                program,
                kernels.writer,
                core,
                with_chunks(work_list_runtime_args(
                    {output_buffer->address(),
                     intermediates_buffer != nullptr ? intermediates_buffer->address() : 0,
                     static_cast<uint32_t>(next.x),
                     static_cast<uint32_t>(next.y),
                     cc.sink ? 1U : 0U},
                    rounds)));
            SetRuntimeArgs(program, kernels.compute_group_1, core, with_chunks(work_list_runtime_args({}, rounds)));
        }
    } else if (use_work_list) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto& rows = work_list.items_per_core[i];
            SetRuntimeArgs(
                program,
                kernels.reader,
                core,
                work_list_runtime_args(
                    {query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     0,  // mask generated on chip
                     gate_buffer != nullptr ? gate_buffer->address() : 0},
                    rows));
            SetRuntimeArgs(
                program,
                kernels.writer,
                core,
                work_list_runtime_args(
                    {output_buffer->address(), intermediates_buffer != nullptr ? intermediates_buffer->address() : 0},
                    rows));
            SetRuntimeArgs(program, kernels.compute_group_1, core, work_list_runtime_args({}, rows));
        }
    } else if (use_balanced_parallelism) {
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
            num_rows_per_core_group_1,
            num_rows_per_core_group_2,
            [&](const CoreWork& work) {
                const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
                SetRuntimeArgs(
                    program,
                    kernels.reader,
                    core,
                    {query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     mask_buffer != nullptr ? mask_buffer->address() : 0,
                     gate_buffer != nullptr ? gate_buffer->address() : 0,
                     num_rows,
                     start_row});
                SetRuntimeArgs(
                    program,
                    kernels.writer,
                    core,
                    {output_buffer->address(),
                     intermediates_buffer != nullptr ? intermediates_buffer->address() : 0,
                     num_rows,
                     start_row});
                SetRuntimeArgs(
                    program, in_group_1 ? kernels.compute_group_1 : kernels.compute_group_2, core, {start_row});
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
