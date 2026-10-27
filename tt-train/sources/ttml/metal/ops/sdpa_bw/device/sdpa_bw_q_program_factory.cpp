// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
const std::string kHasGateDefKey = "HAS_GATE";
const std::string kComputeGradQueryDefKey = "COMPUTE_GRAD_QUERY";
const std::string kComputeUScalerDefKey = "COMPUTE_U_SCALER";
const std::string kComputeGradGateDefKey = "COMPUTE_GRAD_GATE";
const std::string kComputeGatedGradOutputDefKey = "COMPUTE_GATED_GRAD_OUTPUT";

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
                grad_query_buffer != nullptr ? grad_query_buffer->address() : 0U,
                u_scaler_buffer != nullptr ? u_scaler_buffer->address() : 0U,
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
    // Which outputs to produce. Without grad_Q there is no K/V loop: Q, K, V, the mask, the intermediates and the
    // dP/dS/dQ chain are all unused, and u_scaler and the gate outputs only need dY, Y and G.
    const bool compute_grad_query = args.compute_grad_query;
    const bool compute_u_scaler = args.compute_u_scaler;
    const bool compute_grad_gate = args.compute_grad_gate;
    const bool compute_gated_grad_output = args.compute_gated_grad_output;

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
    // Pairing balances the causal K/V loop, which only runs for grad_Q; without it every row costs the same.
    const bool use_balanced_parallelism = compute_grad_query && (args.mask_type == AttentionMaskType::Causal) &&
                                          (St % 2 == 0) && (total_rows_to_process > num_available_cores) && (St > 0);

    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pair_distribution;

    if (use_balanced_parallelism) {
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

    // Y feeds u = rowsum(dY * Y) (grad_Q and the u_scaler output) and grad_gate; the gated dO alone doesn't read it.
    if (compute_grad_query || compute_u_scaler || compute_grad_gate) {
        [[maybe_unused]] auto cb_attn_output = create_circular_buffer(  // CBIndex::c_1
            program,
            all_cores,
            kAttnOutputCbIndex,
            data_format,
            bfloat16_single_tile_size_bytes,
            2 * vWt);
    }

    // Inputs and intermediates of the K/V loop, which only runs for grad_Q.
    if (compute_grad_query) {
        [[maybe_unused]] auto cb_query = create_circular_buffer(  // CBIndex::c_2
            program,
            all_cores,
            kQueryCbIndex,
            data_format,
            bfloat16_single_tile_size_bytes,
            2 * qWt);

        [[maybe_unused]] auto cb_key =  // CBIndex::c_3
            create_circular_buffer(
                program, all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt);

        [[maybe_unused]] auto cb_value =  // CBIndex::c_4
            create_circular_buffer(
                program, all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

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
                kSingleTileBuffer);

        [[maybe_unused]] auto cb_grad_attention =  // CBIndex::c_10
            create_circular_buffer(
                program,
                all_cores,
                kGradAttentionCbIndex,
                precise_data_format,
                float32_single_tile_size_bytes,
                kSingleTileBuffer);

        [[maybe_unused]] auto cb_grad_scores =  // CBIndex::c_11
            create_circular_buffer(
                program,
                all_cores,
                kGradScoresCbIndex,
                precise_data_format,
                float32_single_tile_size_bytes,
                kSingleTileBuffer);

        [[maybe_unused]] auto cb_grad_query =  // CBIndex::c_13
            create_circular_buffer(
                program, all_cores, kGradQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt);
    }

    // u_scaler = rowsum(dY * Y), needed by grad_Q (for its own dS) and by the u_scaler output. compute_u_scalar_row
    // also uses cb_u_scalar_row as scratch for the row reduction, so it exists whenever u_scaler is computed.
    if (compute_grad_query || compute_u_scaler) {
        [[maybe_unused]] auto cb_mat_mul_reduce =  // CBIndex::c_7
            create_circular_buffer(
                program,
                all_cores,
                kMatMulReduceCbIndex,
                precise_data_format,
                float32_single_tile_size_bytes,
                kSingleTileBuffer);

        [[maybe_unused]] auto cb_u_scalar_row =  // CBIndex::c_12
            create_circular_buffer(
                program,
                all_cores,
                kUScalarRowCbIndex,
                precise_data_format,
                float32_single_tile_size_bytes,
                kSingleTileBuffer);
    }

    if (compute_u_scaler) {
        [[maybe_unused]] auto cb_u_scaler_output =  // CBIndex::c_14
            create_circular_buffer(
                program,
                all_cores,
                kUScalerOutputCbIndex,
                precise_data_format,
                float32_single_tile_size_bytes,
                kSingleTileBuffer);
    }

    if (args.has_gate) {
        [[maybe_unused]] auto cb_gate =  // CBIndex::c_15
            create_circular_buffer(
                program, all_cores, kGateCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);

        if (compute_grad_gate) {
            [[maybe_unused]] auto cb_grad_gate =  // CBIndex::c_16
                create_circular_buffer(
                    program, all_cores, kGradGateCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);
        }

        if (compute_gated_grad_output) {
            [[maybe_unused]] auto cb_gated_grad_output =  // CBIndex::c_17
                create_circular_buffer(
                    program, all_cores, kGatedGradOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt);
        }

        // Gated dO as the dP operand, so only for grad_Q.
        if (compute_grad_query) {
            [[maybe_unused]] auto cb_gated_grad_output_local =  // CBIndex::c_18
                create_circular_buffer(
                    program,
                    all_cores,
                    kGatedGradOutputLocalCbIndex,
                    data_format,
                    bfloat16_single_tile_size_bytes,
                    vWt);
        }
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

    // An output is nullopt when it was not requested; its buffer is then null and gets placeholder accessor args
    // and a 0 address, so the writer's accessor offsets do not depend on the flags.
    auto& [grad_query_output, u_scaler_output, grad_gate_output, gated_grad_output_output] = output;
    auto* grad_query_buffer = grad_query_output.has_value() ? grad_query_output.value().buffer() : nullptr;
    auto* u_scaler_buffer = u_scaler_output.has_value() ? u_scaler_output.value().buffer() : nullptr;
    auto* grad_gate_buffer = grad_gate_output.has_value() ? grad_gate_output.value().buffer() : nullptr;
    auto* gated_grad_output_buffer =
        gated_grad_output_output.has_value() ? gated_grad_output_output.value().buffer() : nullptr;

    // Configure defines
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines;

    // Common defines for all kernels
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    // Set defines based on mask type (mutually exclusive). The mask is only applied in the K/V loop, so without
    // grad_Q there is no mask CB and no mask define.
    if (compute_grad_query && args.mask_type == AttentionMaskType::Causal) {
        reader_defines[kCausalMaskDefKey] = "1";
        writer_defines[kCausalMaskDefKey] = "1";
        compute_defines[kCausalMaskDefKey] = "1";
    } else if (compute_grad_query && args.mask_type == AttentionMaskType::Arbitrary) {
        reader_defines[kUseAttnMaskDefKey] = "1";
        compute_defines[kUseAttnMaskDefKey] = "1";
    }

    // Add balanced parallelism define on top of mask type
    if (use_balanced_parallelism) {
        reader_defines[kBalancedParallelismDefKey] = "1";
        writer_defines[kBalancedParallelismDefKey] = "1";
        compute_defines[kBalancedParallelismDefKey] = "1";
    }

    // Gate: reader streams G, compute produces grad_gate + gated dO, writer flushes both
    if (args.has_gate) {
        reader_defines[kHasGateDefKey] = "1";
        writer_defines[kHasGateDefKey] = "1";
        compute_defines[kHasGateDefKey] = "1";
    }

    // Requested outputs
    const auto set_define_on_all = [&](const std::string& key) {
        reader_defines[key] = "1";
        writer_defines[key] = "1";
        compute_defines[key] = "1";
    };
    if (compute_grad_query) {
        set_define_on_all(kComputeGradQueryDefKey);
    }
    if (compute_u_scaler) {
        set_define_on_all(kComputeUScalerDefKey);
    }
    if (compute_grad_gate) {
        set_define_on_all(kComputeGradGateDefKey);
    }
    if (compute_gated_grad_output) {
        set_define_on_all(kComputeGatedGradOutputDefKey);
    }

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    SDPABackwardQKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // 0: Q/K width in tiles
        vWt,              // 1: V/dO/O width in tiles
        St,               // 2: sequence length in tiles
        q_heads,          // 3: number of query heads
        heads_per_group,  // 4: heads per group
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
        qWt,  // 0: query width in tiles
        St,   // 1: sequence length in tiles
        vWt,  // 2: V/dO width in tiles (grad_gate / gated dO rows)
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
            block_size           // 7: block size
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
            block_size                  // 7: block size
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
                block_size                  // 7: block size
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
    if (use_balanced_parallelism) {
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
                    {grad_query_buffer != nullptr ? grad_query_buffer->address() : 0U,
                     u_scaler_buffer != nullptr ? u_scaler_buffer->address() : 0U,
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
    const auto* grad_query_buffer = grad_query_tensor.has_value() ? grad_query_tensor.value().buffer() : nullptr;
    const auto* u_scaler_buffer = u_scaler_tensor.has_value() ? u_scaler_tensor.value().buffer() : nullptr;
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
            runtime_args[kGradQueryBufferIdx] = grad_query_buffer != nullptr ? grad_query_buffer->address() : 0U;
            runtime_args[kUScalerOutputBufferIdx] = u_scaler_buffer != nullptr ? u_scaler_buffer->address() : 0U;
            runtime_args[kGradGateBufferIdx] = grad_gate_buffer != nullptr ? grad_gate_buffer->address() : 0U;
            runtime_args[kGatedGradOutputBufferIdx] =
                gated_grad_output_buffer != nullptr ? gated_grad_output_buffer->address() : 0U;
        }
    });
}

}  // namespace ttml::metal::ops::sdpa_bw::device
