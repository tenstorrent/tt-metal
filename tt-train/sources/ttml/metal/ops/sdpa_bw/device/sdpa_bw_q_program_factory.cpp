// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_q_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <hostdevcommon/common_values.hpp>
#include <string>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/kernel_types.hpp>
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
constexpr uint32_t kGateBufferIdx = 7U;

// Writer runtime args
constexpr uint32_t kGradQueryBufferIdx = 0;
constexpr uint32_t kUScalerOutputBufferIdx = 1U;
constexpr uint32_t kGradGateBufferIdx = 2U;
constexpr uint32_t kGatedGradOutputBufferIdx = 3U;

// Circular buffer indices
constexpr auto kGradOutputCbIndex = tt::CBIndex::c_0;
constexpr auto kAttnOutputCbIndex = tt::CBIndex::c_1;
constexpr auto kQueryCbIndex = tt::CBIndex::c_2;
constexpr auto kKeyCbIndex = tt::CBIndex::c_3;
constexpr auto kValueCbIndex = tt::CBIndex::c_4;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_5;
constexpr auto kIntermediatesCbIndex = tt::CBIndex::c_6;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_7;
constexpr auto kGradQueryAccumCbIndex = tt::CBIndex::c_8;
constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_9;
constexpr auto kGradAttentionCbIndex = tt::CBIndex::c_10;
constexpr auto kGradScoresCbIndex = tt::CBIndex::c_11;
constexpr auto kUScalarRowCbIndex = tt::CBIndex::c_12;
constexpr auto kGradQueryCbIndex = tt::CBIndex::c_13;
constexpr auto kUScalerOutputCbIndex = tt::CBIndex::c_14;
constexpr auto kGateCbIndex = tt::CBIndex::c_15;                  // Input: gate G row
constexpr auto kGradGateCbIndex = tt::CBIndex::c_16;              // Output: grad_gate = dY * Y * (1 - sigmoid(G))
constexpr auto kGatedGradOutputCbIndex = tt::CBIndex::c_17;       // Output: gated dO to DRAM for KV kernel
constexpr auto kGatedGradOutputLocalCbIndex = tt::CBIndex::c_18;  // Compute-local gated dO for dP matmul

constexpr uint32_t kSingleTileBuffer = 1U;
constexpr uint32_t kNumOfIntermCBTiles = 1U;  // single FP32 logsumexp tile per Q row

const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";
const std::string kBalancedParallelismDefKey = "BALANCED_PARALLELISM";
const std::string kWorkListDefKey = "WORK_LIST";
const std::string kChainDefKey = "CHAIN";
constexpr uint32_t kChainSlots = 3U;  // stream chunks resident per core (credits run kChainSlots - 1 ahead)
// Cores per forwarding chain: 7 measured best for the backward at the TP=4 shard shape with the per-round stream
// bounds (iteration 27: 343-346 us against 355 with 14); tunable for sweeps.
const uint32_t kChainLength = env_uint32_or("TTML_SDPA_BW_CHAIN_LENGTH", 7U);
constexpr uint32_t kChainChunk = 4U;      // tile rows of the stream per handshake
constexpr uint32_t kNoRow = 0xFFFFFFFFU;  // "no item this round" in a chain-mode list

// Chain schedule: per stream a contiguous run of cores, the stream's items LPT-scheduled over them, chains of
// kChainLength consecutive cores (highest core injector, lowest core sink). Every core runs `rounds` rounds.
struct ChainCore {
    uint32_t stream = 0;
    bool injector = false;
    bool sink = false;
    uint32_t prev = 0;
    uint32_t next = 0;
    uint32_t chain = 0;  // global chain index
    std::vector<uint32_t> items;
    std::vector<uint32_t> stream_bounds;  // per round: the chain's stream bound (iteration 27), see set_stream_bounds
};

struct ChainSchedule {
    std::vector<ChainCore> cores;
    uint32_t rounds = 0;
    uint32_t num_chains = 0;
};

// Per chain and round the stream bound of the round's items (`bound_of_item`, combined with `combine`, `empty` for a
// round without items): every core of a chain consumes the same chunks, so the chain's items decide (iteration 27).
template <typename BoundFn, typename CombineFn>
void set_stream_bounds(ChainSchedule& sched, BoundFn&& bound_of_item, CombineFn&& combine, uint32_t empty) {
    std::vector<std::vector<uint32_t>> bounds(sched.num_chains, std::vector<uint32_t>(sched.rounds, empty));
    for (const auto& cc : sched.cores) {
        for (uint32_t r = 0; r < cc.items.size(); ++r) {
            bounds[cc.chain][r] = combine(bounds[cc.chain][r], bound_of_item(cc.items[r]));
        }
    }
    for (auto& cc : sched.cores) {
        cc.stream_bounds = bounds[cc.chain];
    }
}

// `items_of_stream(s)` -> (item ids, weights) of stream s.
template <typename ItemsFn>
ChainSchedule chain_schedule(uint32_t num_cores, uint32_t num_streams, ItemsFn&& items_of_stream) {
    ChainSchedule sched;
    sched.cores.resize(num_cores);
    const uint32_t base_cores = num_cores / num_streams;
    const uint32_t extra_cores = num_cores % num_streams;
    uint32_t first_core = 0;
    for (uint32_t stream = 0; stream < num_streams; ++stream) {
        const uint32_t stream_cores = base_cores + (stream < extra_cores ? 1U : 0U);
        auto [items, weights] = items_of_stream(stream);
        const auto lpt = lpt_schedule(weights, stream_cores);
        sched.rounds = std::max(sched.rounds, lpt.max_items);
        for (uint32_t c = 0; c < stream_cores; ++c) {
            auto& cc = sched.cores[first_core + c];
            cc.stream = stream;
            cc.chain = sched.num_chains + c / kChainLength;
            // The chain runs downward in core index (iteration 23): the writer forwards on NOC_1, whose routes
            // decrease in coordinate, so the next core is the previous index (one link) and the injector is the
            // highest core of its chain; the readers' ready counters travel upward on NOC_0.
            const uint32_t pos = c % kChainLength;
            cc.injector = (pos == kChainLength - 1U) || (c + 1U == stream_cores);
            cc.sink = (pos == 0U);
            cc.prev = cc.injector ? first_core + c : first_core + c + 1U;
            cc.next = cc.sink ? first_core + c : first_core + c - 1U;
            for (const uint32_t item : lpt.items_per_core[c]) {
                cc.items.push_back(items[item]);
            }
        }
        first_core += stream_cores;
        sched.num_chains += (stream_cores + kChainLength - 1U) / kChainLength;
    }
    TT_FATAL(
        2U * sched.rounds + 24U <= tt::tt_metal::max_runtime_args,
        "SDPA chain: {} rounds exceed the runtime-arg limit",
        sched.rounds);
    return sched;
}
const std::string kHasGateDefKey = "HAS_GATE";

/**
 * Calculate work-balanced pair distribution for causal SDPA backward.
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
        const uint32_t pairs_for_this_core = pairs_per_core + (core < remainder ? 1 : 0);
        distribution.emplace_back(current_pair, pairs_for_this_core);
        current_pair += pairs_for_this_core;
    }

    return distribution;
}

}  // namespace

namespace ttml::metal::ops::sdpa_bw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct SDPABackwardQKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
/**
 * Set up the runtime arguments for balanced parallelism mode.
 * In this mode, work is distributed by pairs rather than rows.
 * Each pair consists of a "light" row (early in sequence, less work) and
 * a "heavy" row (late in sequence, more work), providing balanced total work.
 */
void assign_per_core_runtime_args_balanced(
    tt::tt_metal::Program& program,
    const SDPABackwardQKernels& kernels,
    const tt::tt_metal::Buffer* grad_output_buffer,
    const tt::tt_metal::Buffer* attn_output_buffer,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    const tt::tt_metal::Buffer* gate_buffer,
    const tt::tt_metal::Buffer* grad_gate_buffer,
    const tt::tt_metal::Buffer* gated_grad_output_buffer,
    const tt::tt_metal::Buffer* grad_query_buffer,
    const tt::tt_metal::Buffer* u_scaler_buffer,
    const uint32_t num_cores,
    const uint32_t num_cores_y,
    const std::vector<std::pair<uint32_t, uint32_t>>& pair_distribution) {
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& [start_pair_idx, num_pairs] = pair_distribution[i];

        // Reader kernel: reuses num_rows_to_process = num_pairs, start_row = start_pair_idx
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {
                grad_output_buffer->address(),
                attn_output_buffer->address(),
                query_buffer->address(),
                key_buffer->address(),
                value_buffer->address(),
                0U,  // mask_addr unused for balanced causal
                intermediates_buffer->address(),
                gate_buffer != nullptr ? gate_buffer->address() : 0U,  // gate buffer address
                num_pairs,
                start_pair_idx,
            });

        // Writer kernel: num_rows_to_process = num_pairs, start_row = start_pair_idx
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {
                grad_query_buffer->address(),
                u_scaler_buffer->address(),
                grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U,  // grad_gate buffer address
                gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address()
                                                    : 0U,  // gated dO buffer address
                num_pairs,
                start_pair_idx,
            });

        // Compute kernel: (start_pair_idx, num_pairs)
        SetRuntimeArgs(program, kernels.compute_group_1, core, {start_pair_idx, num_pairs});
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
    const auto& gate = tensor_args.gate;

    auto* device = grad_output.device();
    tt::tt_metal::Program program{};
    auto input_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // Get tensor dimensions and extract heads from shapes
    const auto [qB, qNH, qS, qEmbd] = query.padded_shape().to_array_4D();
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

    const uint32_t qWt = qEmbd / tt::constants::TILE_WIDTH;  // num of tiles in Q/K inner dim
    const uint32_t kWt = kEmbd / tt::constants::TILE_WIDTH;
    const uint32_t vWt = vEmbd / tt::constants::TILE_WIDTH;  // num of tiles in V/dO/O inner dim

    // Scale factor from Q/K dimension (not V dimension)
    const float per_head_dim = static_cast<float>(qEmbd);
    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(per_head_dim));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_available_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // -------------------------------------------------------------------------
    // Balanced Parallelism Decision
    // -------------------------------------------------------------------------
    const uint32_t pairs_per_seq = St / 2;
    const uint32_t total_pairs = NC * pairs_per_seq;
    // Work-list mode (causal): host LPT schedule of the rows, per-core row lists as runtime args (see sdpa_fw).
    // Chain mode (causal): the K/V stream of a (batch, kv group) is read once per chain of cores and forwarded
    // core to core (see sdpa_fw); the stream's rows are LPT-scheduled over the stream's cores.
    const bool causal = (args.mask_type == AttentionMaskType::Causal) && (St > 0);
    const uint32_t num_streams = qB * kv_heads;
    const uint32_t chain_chunk = (St % kChainChunk == 0U) ? kChainChunk : ((St % 2U == 0U) ? 2U : 1U);
    // TTML_SDPA_BW_Q_STREAM=list: work lists without the K/V forwarding chain (attribution experiments).
    const char* stream_mode = std::getenv("TTML_SDPA_BW_Q_STREAM");
    const bool use_chain = causal && (num_available_cores >= 2U * num_streams) &&
                           !(stream_mode != nullptr && std::string(stream_mode) == "list");
    const bool use_work_list = causal && !use_chain;
    const bool use_balanced_parallelism = false;

    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pair_distribution;
    LptSchedule work_list;
    ChainSchedule chain;

    const auto items_of_stream = [&](uint32_t stream) {
        const uint32_t b = stream / kv_heads;
        const uint32_t kv = stream % kv_heads;
        std::vector<uint32_t> items;
        std::vector<uint32_t> weights;
        for (uint32_t h = kv * heads_per_group; h < (kv + 1U) * heads_per_group; ++h) {
            for (uint32_t q = 0; q < St; ++q) {
                items.push_back((b * qNH + h) * St + q);
                weights.push_back(q + 2U);
            }
        }
        return std::make_pair(items, weights);
    };
    if (use_chain) {
        num_cores = std::min(num_available_cores, total_rows_to_process);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, false);
        core_group_1 = all_cores;
        chain = chain_schedule(num_cores, num_streams, items_of_stream);
        // The stream stops at the diagonal chunk of the round's heaviest row.
        set_stream_bounds(
            chain,
            [&](uint32_t item) { return ((item % St) + 1U + chain_chunk - 1U) / chain_chunk; },
            [](uint32_t a, uint32_t b) { return std::max(a, b); },
            /* empty */ 0U);
        num_rows_per_core_group_1 = chain.rounds;
    } else if (use_work_list) {
        num_cores = std::min(num_available_cores, total_rows_to_process);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, false);
        core_group_1 = all_cores;
        // Row r of a sequence processes r + 1 K/V tiles plus a constant (its Q/dO/O reads, u_scalar, dQ write).
        std::vector<uint32_t> weights(total_rows_to_process);
        for (uint32_t row = 0; row < total_rows_to_process; ++row) {
            weights[row] = (row % St) + 2U;
        }
        work_list = lpt_schedule(weights, num_cores);
        TT_FATAL(
            work_list.max_items + 16U <= tt::tt_metal::max_runtime_args,
            "SDPA work list: a core would receive {} items, more than the runtime-arg limit allows",
            work_list.max_items);
        num_rows_per_core_group_1 = work_list.max_items;
    } else if (use_balanced_parallelism) {
        num_cores = std::min(num_available_cores, total_pairs);
        all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, false);
        core_group_1 = all_cores;

        pair_distribution = calculate_balanced_pair_distribution(total_pairs, num_cores);
        num_rows_per_core_group_1 = pair_distribution.empty() ? 0 : pair_distribution[0].second * 2;
    } else {
        auto work_split = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
        num_cores = std::get<0>(work_split);
        all_cores = std::get<1>(work_split);
        core_group_1 = std::get<2>(work_split);
        core_group_2 = std::get<3>(work_split);
        num_rows_per_core_group_1 = std::get<4>(work_split);
        num_rows_per_core_group_2 = std::get<5>(work_split);
    }

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    const uint32_t block_size = get_block_size(qWt, 4U);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_grad_output = create_circular_buffer(  // CBIndex::c_0
        program,
        all_cores,
        kGradOutputCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        2 * vWt);

    [[maybe_unused]] auto cb_attn_output = create_circular_buffer(  // CBIndex::c_1
        program,
        all_cores,
        kAttnOutputCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        2 * vWt);

    [[maybe_unused]] auto cb_query = create_circular_buffer(  // CBIndex::c_2
        program,
        all_cores,
        kQueryCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        2 * qWt);

    // The compute kernel works on K/V blocks of kv_block rows: the chain chunk under CHAIN (the reader pushes and
    // the compute pops whole chunks, so a block never straddles the CB end); one row in the other modes, whose
    // readers push row by row (review round 9 P1: a multi-row block there walked past the CB end after a
    // partial last block). Two blocks resident, and the P / dP / dS CBs hold one block each.
    const uint32_t kv_block = use_chain ? chain_chunk : 1U;
    const uint32_t kv_cb_rows = (use_chain ? kChainSlots : 2U) * kv_block;
    [[maybe_unused]] auto cb_key =  // CBIndex::c_3
        create_circular_buffer(
            program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, kv_cb_rows * kWt);

    [[maybe_unused]] auto cb_value =  // CBIndex::c_4
        create_circular_buffer(
            program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, kv_cb_rows * vWt);

    // Create mask buffer if using attention mask from DRAM or generating causal mask on-the-fly
    // Not needed for AttentionMaskType::None
    if (args.mask_type != AttentionMaskType::None) {
        [[maybe_unused]] auto cb_attn_mask =  // CBIndex::c_5
            create_circular_buffer(
                program,
                all_cores,
                kAttnMaskCbIndex,
                data_format,
                bfloat16_single_tile_size_bytes,
                2 * kSingleTileBuffer);
    }

    [[maybe_unused]] auto cb_intermediates =  // CBIndex::c_6
        create_circular_buffer(
            program,
            all_cores,
            kIntermediatesCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            2 * kNumOfIntermCBTiles);

    [[maybe_unused]] auto cb_mat_mul_reduce =  // CBIndex::c_7
        create_circular_buffer(
            program,
            all_cores,
            kMatMulReduceCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_query_accum =  // CBIndex::c_8
        create_circular_buffer(
            program, all_cores, kGradQueryAccumCbIndex, precise_data_format, float32_single_tile_size_bytes, qWt);

    [[maybe_unused]] auto cb_attention_weights =  // CBIndex::c_9
        create_circular_buffer(
            program,
            all_cores,
            kAttentionWeightsCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kv_block);

    [[maybe_unused]] auto cb_grad_attention =  // CBIndex::c_10
        create_circular_buffer(
            program, all_cores, kGradAttentionCbIndex, precise_data_format, float32_single_tile_size_bytes, kv_block);

    [[maybe_unused]] auto cb_grad_scores =  // CBIndex::c_11
        create_circular_buffer(
            program, all_cores, kGradScoresCbIndex, precise_data_format, float32_single_tile_size_bytes, kv_block);

    [[maybe_unused]] auto cb_u_scaler_row =  // CBIndex::c_12
        create_circular_buffer(
            program,
            all_cores,
            kUScalarRowCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    [[maybe_unused]] auto cb_grad_query =  // CBIndex::c_13
        create_circular_buffer(
            program, all_cores, kGradQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);

    [[maybe_unused]] auto cb_u_scaler_output =  // CBIndex::c_14
        create_circular_buffer(
            program,
            all_cores,
            kUScalerOutputCbIndex,
            precise_data_format,
            float32_single_tile_size_bytes,
            kSingleTileBuffer);

    if (args.has_gate) {
        [[maybe_unused]] auto cb_gate =  // CBIndex::c_15
            create_circular_buffer(
                program, all_cores, kGateCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

        [[maybe_unused]] auto cb_grad_gate =  // CBIndex::c_16
            create_circular_buffer(
                program, all_cores, kGradGateCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

        [[maybe_unused]] auto cb_gated_grad_output =  // CBIndex::c_17
            create_circular_buffer(
                program, all_cores, kGatedGradOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

        [[maybe_unused]] auto cb_gated_grad_output_local =  // CBIndex::c_18
            create_circular_buffer(
                program, all_cores, kGatedGradOutputLocalCbIndex, data_format, bfloat16_single_tile_size_bytes, vWt);
    }

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

    auto* gate_buffer = args.has_gate ? gate.value().buffer() : nullptr;

    auto& [grad_query_output, u_scaler_output, grad_gate_output, gated_grad_output_output] = output;
    auto* grad_query_buffer = grad_query_output.buffer();
    auto* u_scaler_buffer = u_scaler_output.buffer();
    auto* grad_gate_buffer = args.has_gate ? grad_gate_output.value().buffer() : nullptr;
    auto* gated_grad_output_buffer = args.has_gate ? gated_grad_output_output.value().buffer() : nullptr;

    // Configure defines
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines;

    // Common defines for all kernels
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    // Set defines based on mask type (mutually exclusive)
    if (args.mask_type == AttentionMaskType::Causal) {
        reader_defines[kCausalMaskDefKey] = "1";
        writer_defines[kCausalMaskDefKey] = "1";
        compute_defines[kCausalMaskDefKey] = "1";
    } else if (args.mask_type == AttentionMaskType::Arbitrary) {
        reader_defines[kUseAttnMaskDefKey] = "1";
        compute_defines[kUseAttnMaskDefKey] = "1";
    }

    // Add balanced parallelism define on top of mask type
    if (use_balanced_parallelism) {
        reader_defines[kBalancedParallelismDefKey] = "1";
        writer_defines[kBalancedParallelismDefKey] = "1";
        compute_defines[kBalancedParallelismDefKey] = "1";
    }
    if (use_work_list || use_chain) {
        reader_defines[kWorkListDefKey] = "1";
        writer_defines[kWorkListDefKey] = "1";
        compute_defines[kWorkListDefKey] = "1";
    }
    if (use_chain) {
        reader_defines[kChainDefKey] = "1";
        writer_defines[kChainDefKey] = "1";
        reader_defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);
        writer_defines["CHAIN_SLOTS"] = std::to_string(kChainSlots);
        compute_defines[kChainDefKey] = "1";
    }
    const uint32_t chain_sender_sem = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    const uint32_t chain_receiver_sem = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // The VALID relay of iterations 15-25 was replaced by the counting receiver semaphore (iteration 26); its
    // compile-arg slot stays at 0 so the argument indices after it are unchanged.
    constexpr uint32_t chain_valid_sem = 0U;
    const uint32_t chain_received_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);   // reader -> writer
    const uint32_t chain_forwarded_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);  // writer -> reader

    // Gate: reader streams G, compute produces grad_gate + gated dO, writer flushes both
    if (args.has_gate) {
        reader_defines[kHasGateDefKey] = "1";
        writer_defines[kHasGateDefKey] = "1";
        compute_defines[kHasGateDefKey] = "1";
    }

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    SDPABackwardQKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,                  // 0: Q/K width in tiles
        vWt,                  // 1: V/dO/O width in tiles
        St,                   // 2: sequence length in tiles
        q_heads,              // 3: number of query heads
        heads_per_group,      // 4: heads per group
        chain_sender_sem,     // 5: chain mode semaphores
        chain_receiver_sem,   // 6
        chain_valid_sem,      // 7
        chain_chunk,          // 8: chain mode K/V rows per handshake
        chain_received_sem,   // 9: chain mode: chunks the reader has received (to the writer)
        chain_forwarded_sem,  // 10: chain mode: chunks the writer has forwarded (to the reader)
    };
    tt::tt_metal::TensorAccessorArgs(grad_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(attn_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(attn_mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, reader_defines, kReaderKernelPath);

    // Writer compile-time arguments
    std::vector<uint32_t> writer_compile_args = {
        qWt,                  // 0: query width in tiles
        St,                   // 1: sequence length in tiles
        vWt,                  // 2: V/dO width in tiles (grad_gate / gated dO rows)
        chain_chunk,          // 3: chain mode K/V rows per handshake
        chain_sender_sem,     // 4: chain mode semaphores (the writer forwards the stream)
        chain_receiver_sem,   // 5
        chain_valid_sem,      // 6
        chain_received_sem,   // 7
        chain_forwarded_sem,  // 8
    };
    tt::tt_metal::TensorAccessorArgs(grad_query_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(u_scaler_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(grad_gate_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(gated_grad_output_buffer).append_to(writer_compile_args);

    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, writer_defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // Set UnpackToDestFp32 for buffers that must preserve full FP32 in DST registers.
    // These buffers use unpack-to-dest + SFPU path, never FPU (which truncates to TF32).
    auto create_unpack_to_dest_mode = []() {
        std::vector<tt::tt_metal::UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        mode[tt::CBIndex::c_8] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // kGradQueryAccumCbIndex
        mode[tt::CBIndex::c_9] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // kAttentionWeightsCbIndex
        return mode;
    };

    if (use_balanced_parallelism) {
        const uint32_t max_pairs_per_core = pair_distribution.empty() ? 0 : pair_distribution[0].second;

        std::vector<uint32_t> compute_args = {
            max_pairs_per_core,  // 0: num_pairs (max pairs per core)
            qWt,                 // 1: Q/K inner dim in tiles
            vWt,                 // 2: V/dO/O inner dim in tiles
            St,                  // 3: num_seq_len / TILE_H
            scaler,              // 4: sqrt(Et) - sdpa scaler factor
            minus_one,           // 5: used to transform mask from 1/0 to 0/-1
            custom_inf,          // 6: used to transform mask from 0/-1 to 0/-inf
            block_size,          // 7: block size
            kv_block             // 8: K/V rows per compute block (the chain chunk under CHAIN)
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
                .defines = compute_defines});
    } else {
        // Group 1 compile-time arguments
        std::vector<uint32_t> compute_group_1_args = {
            num_rows_per_core_group_1,  // 0: per_core_block_cnt
            qWt,                        // 1: Q/K inner dim in tiles
            vWt,                        // 2: V/dO/O inner dim in tiles
            St,                         // 3: num_seq_len / TILE_H
            scaler,                     // 4: sqrt(Et) - sdpa scaler factor
            minus_one,                  // 5: used to transform mask from 1/0 to 0/-1
            custom_inf,                 // 6: used to transform mask from 0/-1 to 0/-inf
            block_size,                 // 7: block size
            kv_block                    // 8: K/V rows per compute block (the chain chunk under CHAIN)
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
                .defines = compute_defines});

        // Group 2 compile-time arguments
        if (!core_group_2.ranges().empty()) {
            std::vector<uint32_t> compute_group_2_args = {
                num_rows_per_core_group_2,  // 0: per_core_block_cnt
                qWt,                        // 1: Q/K inner dim in tiles
                vWt,                        // 2: V/dO/O inner dim in tiles
                St,                         // 3: num_seq_len / TILE_H
                scaler,                     // 4: sqrt(Et) - sdpa scaler factor
                minus_one,                  // 5: used to transform mask from 1/0 to 0/-1
                custom_inf,                 // 6: used to transform mask from 0/-1 to 0/-inf
                block_size,                 // 7: block size
                kv_block                    // 8: K/V rows per compute block (the chain chunk under CHAIN)
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
                    .defines = compute_defines});
        }
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    if (use_chain) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto& cc = chain.cores[i];
            std::vector<uint32_t> rounds(cc.items);
            rounds.resize(chain.rounds, kNoRow);
            // The items (padded) then, per round, the chain's stream bound.
            const auto with_bounds = [&](std::vector<uint32_t> args) {
                args.insert(args.end(), cc.stream_bounds.begin(), cc.stream_bounds.end());
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
                with_bounds(work_list_runtime_args(
                    {grad_output_buffer->address(),
                     attn_output_buffer->address(),
                     query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     0U,  // mask generated on chip
                     intermediates_buffer->address(),
                     gate_buffer != nullptr ? gate_buffer->address() : 0U,
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
                with_bounds(work_list_runtime_args(
                    {grad_query_buffer->address(),
                     u_scaler_buffer->address(),
                     grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U,
                     gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address() : 0U,
                     static_cast<uint32_t>(next.x),
                     static_cast<uint32_t>(next.y),
                     cc.sink ? 1U : 0U},
                    rounds)));
            SetRuntimeArgs(program, kernels.compute_group_1, core, with_bounds(work_list_runtime_args({}, rounds)));
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
                    {grad_output_buffer->address(),
                     attn_output_buffer->address(),
                     query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     0U,  // mask generated on chip
                     intermediates_buffer->address(),
                     gate_buffer != nullptr ? gate_buffer->address() : 0U},
                    rows));
            SetRuntimeArgs(
                program,
                kernels.writer,
                core,
                work_list_runtime_args(
                    {grad_query_buffer->address(),
                     u_scaler_buffer->address(),
                     grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U,
                     gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address() : 0U},
                    rows));
            SetRuntimeArgs(program, kernels.compute_group_1, core, work_list_runtime_args({}, rows));
        }
    } else if (use_balanced_parallelism) {
        assign_per_core_runtime_args_balanced(
            program,
            kernels,
            grad_output_buffer,
            attn_output_buffer,
            query_buffer,
            key_buffer,
            value_buffer,
            intermediates_buffer,
            gate_buffer,
            grad_gate_buffer,
            gated_grad_output_buffer,
            grad_query_buffer,
            u_scaler_buffer,
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
                    {grad_output_buffer->address(),
                     attn_output_buffer->address(),
                     query_buffer->address(),
                     key_buffer->address(),
                     value_buffer->address(),
                     attn_mask_buffer != nullptr ? attn_mask_buffer->address() : 0U,
                     intermediates_buffer->address(),
                     gate_buffer != nullptr ? gate_buffer->address() : 0U,
                     num_rows,
                     start_row});
                SetRuntimeArgs(
                    program,
                    kernels.writer,
                    core,
                    {grad_query_buffer->address(),
                     u_scaler_buffer->address(),
                     grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U,
                     gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address() : 0U,
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
    const auto* gate_buffer = tensor_args.gate.has_value() ? tensor_args.gate.value().buffer() : nullptr;

    auto& [grad_query_tensor, u_scaler_tensor, grad_gate_tensor, gated_grad_output_tensor] = tensor_return_value;
    const auto* grad_query_buffer = grad_query_tensor.buffer();
    const auto* u_scaler_buffer = u_scaler_tensor.buffer();
    const auto* grad_gate_buffer = grad_gate_tensor.has_value() ? grad_gate_tensor.value().buffer() : nullptr;
    const auto* gated_grad_output_buffer =
        gated_grad_output_tensor.has_value() ? gated_grad_output_tensor.value().buffer() : nullptr;

    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_bw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_bw_writer_kernel);

    for_each_core(num_cores, num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
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
            runtime_args[kGateBufferIdx] = gate_buffer != nullptr ? gate_buffer->address() : 0U;
        }

        // Update output buffers for writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kGradQueryBufferIdx] = grad_query_buffer->address();
            runtime_args[kUScalerOutputBufferIdx] = u_scaler_buffer->address();
            runtime_args[kGradGateBufferIdx] = grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U;
            runtime_args[kGatedGradOutputBufferIdx] =
                gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address() : 0U;
        }
    });
}

}  // namespace ttml::metal::ops::sdpa_bw::device
