// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_kv_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "sdpa_bw_kv_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_reader_kernel.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_writer_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_kv_compute_kernel.cpp";

// Reader runtime args (fixed order — u_scaler always present, attn_output removed)
constexpr uint32_t kReaderGradOutputIdx = 0;
constexpr uint32_t kReaderQueryIdx = 1U;
constexpr uint32_t kReaderKeyIdx = 2U;
constexpr uint32_t kReaderValueIdx = 3U;
constexpr uint32_t kReaderMaskIdx = 4U;
constexpr uint32_t kReaderIntermediatesIdx = 5U;
constexpr uint32_t kReaderUScalerIdx = 6U;

// Writer runtime args
constexpr uint32_t kGradKeyBufferIdx = 0;
constexpr uint32_t kGradValueBufferIdx = 1U;

// Kernel handles in the built Program: the push order of ProgramDescriptor::kernels. The group 2 compute kernel (3)
// exists only for an uneven row split; override_runtime_arguments() touches only the reader and the writer.
constexpr uint32_t kReaderKernel = 0U;
constexpr uint32_t kWriterKernel = 1U;
constexpr uint32_t kComputeGroup1Kernel = 2U;
constexpr uint32_t kComputeGroup2Kernel = 3U;

// Circular buffer indices
constexpr auto kGradOutputCbIndex = tt::CBIndex::c_0;
constexpr auto kQueryCbIndex = tt::CBIndex::c_2;
constexpr auto kKeyCbIndex = tt::CBIndex::c_3;
constexpr auto kValueCbIndex = tt::CBIndex::c_4;
constexpr auto kAttnMaskCbIndex = tt::CBIndex::c_5;
constexpr auto kIntermediatesCbIndex = tt::CBIndex::c_6;
constexpr auto kMatMulReduceCbIndex = tt::CBIndex::c_7;
constexpr auto kGradValueAccumCbIndex = tt::CBIndex::c_8;
constexpr auto kGradKeyAccumCbIndex = tt::CBIndex::c_9;
constexpr auto kAttentionWeightsCbIndex = tt::CBIndex::c_10;
constexpr auto kGradAttentionCbIndex = tt::CBIndex::c_11;
constexpr auto kGradScoresTransposedCbIndex = tt::CBIndex::c_12;   // dS^T, packed directly from DST
constexpr auto kAttnWeightsTransposedCbIndex = tt::CBIndex::c_13;  // P^T, packed directly from DST
constexpr auto kUScalarRowCbIndex = tt::CBIndex::c_14;
constexpr auto kGradKeyCbIndex = tt::CBIndex::c_15;
constexpr auto kGradValueCbIndex = tt::CBIndex::c_16;

constexpr uint32_t kSingleTileBuffer = 1U;
constexpr uint32_t kNumOfIntermCBTiles = 1U;  // single FP32 logsumexp tile per Q row

const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";
const std::string kBalancedParallelismDefKey = "BALANCED_PARALLELISM";

/**
 * Calculate work-balanced pair distribution for causal SDPA backward KV.
 *
 * For causal attention, K/V row k processes (Ht-k) Q tiles.
 * By pairing row k with row (Ht-1-k), each pair has constant work = Ht+1.
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

// How the rows split over the cores: for causal attention with an even number of sequence tiles and more rows than
// cores, pairs of a light and a heavy row (balanced parallelism), else split_work_to_cores' row split. Either way the
// cores with work are the first `num_cores` of the walk, core i being {i / num_cores_y, i % num_cores_y}.
struct RowSplit {
    bool balanced = false;
    uint32_t num_cores = 0U;
};

RowSplit choose_row_split(
    ttml::metal::AttentionMaskType mask_type, uint32_t St, uint32_t NC, uint32_t num_available_cores) {
    const uint32_t total_rows = NC * St;
    const bool balanced = (mask_type == ttml::metal::AttentionMaskType::Causal) && (St % 2 == 0) &&
                          (total_rows > num_available_cores) && (St > 0);
    const uint32_t units = balanced ? NC * (St / 2U) : total_rows;
    return RowSplit{.balanced = balanced, .num_cores = std::min(num_available_cores, units)};
}

// An optional buffer's runtime arg: the buffer when present, 0 when not.
void push_buffer_or_zero(tt::tt_metal::KernelDescriptor::RTArgList& args, tt::tt_metal::Buffer* buffer) {
    if (buffer != nullptr) {
        args.push_back(buffer);
    } else {
        args.push_back(0U);
    }
}

}  // namespace

namespace ttml::metal::ops::sdpa_bw::device {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SDPABackwardKVDeviceOperation, SDPABackwardKVProgramFactory);

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
/**
 * Set up the runtime arguments for balanced parallelism mode.
 * Work is distributed by pairs rather than rows.
 * Each pair consists of a "light" K/V row (late in sequence, less Q work) and
 * a "heavy" K/V row (early in sequence, more Q work), providing balanced total work.
 */
void assign_per_core_runtime_args_balanced(
    tt::tt_metal::ProgramDescriptor& desc,
    tt::tt_metal::Buffer* grad_output_buffer,
    tt::tt_metal::Buffer* query_buffer,
    tt::tt_metal::Buffer* key_buffer,
    tt::tt_metal::Buffer* value_buffer,
    tt::tt_metal::Buffer* intermediates_buffer,
    tt::tt_metal::Buffer* u_scaler_buffer,
    tt::tt_metal::Buffer* grad_key_buffer,
    tt::tt_metal::Buffer* grad_value_buffer,
    const uint32_t num_cores,
    const uint32_t num_cores_y,
    const std::vector<std::pair<uint32_t, uint32_t>>& pair_distribution) {
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& [start_pair_idx, num_pairs] = pair_distribution[i];

        tt::tt_metal::KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(grad_output_buffer);
        reader_args.push_back(query_buffer);
        reader_args.push_back(key_buffer);
        reader_args.push_back(value_buffer);
        reader_args.push_back(0U);  // mask_addr unused for balanced causal
        reader_args.push_back(intermediates_buffer);
        reader_args.push_back(u_scaler_buffer);
        reader_args.push_back(num_pairs);
        reader_args.push_back(start_pair_idx);
        desc.kernels[kReaderKernel].emplace_runtime_args(core, reader_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(grad_key_buffer);
        writer_args.push_back(grad_value_buffer);
        writer_args.push_back(num_pairs);
        writer_args.push_back(start_pair_idx);
        desc.kernels[kWriterKernel].emplace_runtime_args(core, writer_args);

        desc.kernels[kComputeGroup1Kernel].emplace_runtime_args(core, {start_pair_idx, num_pairs});
    }
}

tt::tt_metal::ProgramDescriptor SDPABackwardKVProgramFactory::create_descriptor(
    const kv::operation_attributes_t& args, const kv::tensor_args_t& tensor_args, kv::tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& grad_output = tensor_args.grad_output;
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& intermediates = tensor_args.intermediates;

    auto* device = grad_output.device();

    tt::tt_metal::ProgramDescriptor desc;
    auto input_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float32);

    // Get tensor dimensions and extract heads from shapes
    const auto [qB, qNH, qS, qEmbd] = query.padded_shape().to_array_4D();
    const auto [kB, kNH, kS, kEmbd] = key.padded_shape().to_array_4D();
    const auto [vB, vNH, vS, vEmbd] = value.padded_shape().to_array_4D();

    // For backward pass we split work over rows of K and V
    // Each row corresponds to a group in K and V, and all associated heads in Q
    // TODO[improvement](vmelnykov): explore splitting work over cores using assumption that attn_mask is
    // causal(triangular).
    const uint32_t St = qS / tt::constants::TILE_HEIGHT;  // num of tiles in seq len dim
    const uint32_t NC = kB * kNH;
    const uint32_t total_rows_to_process =
        NC * St;                    // total rows to process = batch_size * num_heads * num_tiles_in_seq_len
    const uint32_t kv_heads = kNH;  // number of heads in Key and Value
    const uint32_t heads_per_group = qNH / kv_heads;  // we read heads_per_group heads from Q for one group of K and V
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
    const RowSplit row_split = choose_row_split(args.mask_type, St, NC, num_available_cores);
    const bool use_balanced_parallelism = row_split.balanced;

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

    const uint32_t block_size_q = get_block_size(qWt, 4U);
    const uint32_t block_size_v = get_block_size(vWt, 4U);

    const auto data_format = input_data_format;
    const auto precise_data_format = tt::DataFormat::Float32;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGradOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kQueryCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * qWt));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt));

    // Create mask buffer if using attention mask from DRAM or generating causal mask on-the-fly
    // Not needed for AttentionMaskType::None
    if (args.mask_type != AttentionMaskType::None) {
        desc.cbs.push_back(make_cb_descriptor(
            all_cores, kAttnMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kSingleTileBuffer));
    }

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kIntermediatesCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        2 * kNumOfIntermCBTiles));

    // Utility buffers
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kMatMulReduceCbIndex, precise_data_format, float32_single_tile_size_bytes, kSingleTileBuffer));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGradValueAccumCbIndex, precise_data_format, float32_single_tile_size_bytes, vWt));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGradKeyAccumCbIndex, precise_data_format, float32_single_tile_size_bytes, kWt));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kAttnWeightsTransposedCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kSingleTileBuffer));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kAttentionWeightsCbIndex, precise_data_format, float32_single_tile_size_bytes, kSingleTileBuffer));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kGradAttentionCbIndex, precise_data_format, float32_single_tile_size_bytes, kSingleTileBuffer));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kGradScoresTransposedCbIndex,
        precise_data_format,
        float32_single_tile_size_bytes,
        kSingleTileBuffer));

    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kUScalarRowCbIndex, precise_data_format, float32_single_tile_size_bytes, kSingleTileBuffer));

    // Output gradient buffers (grad_query is computed by Q kernel, not KV kernel)
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGradKeyCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * kWt));

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kGradValueCbIndex, data_format, bfloat16_single_tile_size_bytes, 2 * vWt));

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* grad_output_buffer = grad_output.buffer();
    auto* query_buffer = query.buffer();
    auto* key_buffer = key.buffer();
    auto* value_buffer = value.buffer();
    auto* mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask.value().buffer() : nullptr;
    auto* intermediates_buffer = intermediates.buffer();
    auto* u_scaler_buffer = tensor_args.u_scaler.buffer();

    auto& [grad_key, grad_value] = output;
    auto* grad_key_buffer = grad_key.buffer();
    auto* grad_value_buffer = grad_value.buffer();

    // Configure defines
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines;

    // Common defines for all kernels
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    // Mask type defines
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

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        qWt,              // 0: Q/K width in tiles (Q@K^T path)
        kWt,              // 1: key width in tiles (K read)
        vWt,              // 2: value/dO/O width in tiles
        St,               // 3: sequence length in tiles
        qNH,              // 4: number of query heads
        heads_per_group,  // 5: heads per group
    };
    tt::tt_metal::TensorAccessorArgs(grad_output_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(u_scaler_buffer).append_to(reader_compile_args);

    // Kernels are pushed in the order of the k*Kernel handles: reader, writer, then the compute kernels.
    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_args, reader_defines, kReaderKernelPath));

    // Writer compile-time arguments
    std::vector<uint32_t> writer_compile_args = {
        kWt,             // 0: key width in tiles (grad_K)
        vWt,             // 1: value width in tiles (grad_V)
        St,              // 2: sequence length in tiles
        qNH,             // 3: number of query heads
        heads_per_group  // 4: heads per group
    };
    tt::tt_metal::TensorAccessorArgs(grad_key_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(grad_value_buffer).append_to(writer_compile_args);

    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_args, writer_defines, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    auto create_unpack_to_dest_mode = []() {
        std::vector<tt::tt_metal::UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        mode[tt::CBIndex::c_8] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // kGradValueAccumCbIndex
        mode[tt::CBIndex::c_9] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // kGradKeyAccumCbIndex
        // P (c_10) is consumed only via copy-to-dest in the compute kernel (never through a
        // Src-register unpack), so unpack-to-dest keeps it at full FP32 through the
        // elementwise dS chain. dS never leaves DST untransposed: P^T and dS^T are produced
        // by in-DST transpose_dest and packed directly to c_13/c_12, which stay Default —
        // they feed matmul Src registers, where Float32 unpack is not supported.
        mode[tt::CBIndex::c_10] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // kAttentionWeightsCbIndex
        return mode;
    };

    if (use_balanced_parallelism) {
        const uint32_t max_pairs_per_core = pair_distribution.empty() ? 0 : pair_distribution[0].second;

        std::vector<uint32_t> compute_args = {
            max_pairs_per_core,  // 0: num_pairs (max pairs per core)
            block_size_q,        // 1: block size for Q/K matmuls (divides qWt)
            block_size_v,        // 2: block size for V/dO matmuls (divides vWt)
            qWt,                 // 3: Q/K inner dim in tiles
            vWt,                 // 4: V/dO/O inner dim in tiles
            St,                  // 5: sequence length in tiles
            heads_per_group,     // 6: heads per group
            scaler,              // 7: scale factor
            minus_one,           // 8: mask transform constant
            custom_inf           // 9: used to transform mask from 0/-1 to 0/-inf
        };

        desc.kernels.push_back(make_compute_kernel_descriptor(
            all_cores,
            compute_args,
            compute_defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true,
            create_unpack_to_dest_mode(),
            math_fidelity,
            /*math_approx_mode=*/false));
    } else {
        // Group 1 compile-time arguments
        std::vector<uint32_t> compute_group_1_args = {
            num_rows_per_core_group_1,  // 0: per_core_block_cnt
            block_size_q,               // 1: block size for Q/K matmuls (divides qWt)
            block_size_v,               // 2: block size for V/dO matmuls (divides vWt)
            qWt,                        // 3: Q/K inner dim in tiles
            vWt,                        // 4: V/dO/O inner dim in tiles
            St,                         // 5: sequence length in tiles
            heads_per_group,            // 6: heads per group
            scaler,                     // 7: scale factor
            minus_one,                  // 8: mask transform constant
            custom_inf                  // 9: used to transform mask from 0/-1 to 0/-inf
        };

        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_1,
            compute_group_1_args,
            compute_defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true,
            create_unpack_to_dest_mode(),
            math_fidelity,
            /*math_approx_mode=*/false));

        // Group 2 (if present)
        if (!core_group_2.ranges().empty()) {
            std::vector<uint32_t> compute_group_2_args = {
                num_rows_per_core_group_2,  // 0: per_core_block_cnt
                block_size_q,               // 1: block size for Q/K matmuls (divides qWt)
                block_size_v,               // 2: block size for V/dO matmuls (divides vWt)
                qWt,                        // 3: Q/K inner dim in tiles
                vWt,                        // 4: V/dO/O inner dim in tiles
                St,                         // 5: sequence length in tiles
                heads_per_group,            // 6: heads per group
                scaler,                     // 7: scale factor
                minus_one,                  // 8: mask transform constant
                custom_inf                  // 9: used to transform mask from 0/-1 to 0/-inf
            };

            desc.kernels.push_back(make_compute_kernel_descriptor(
                core_group_2,
                compute_group_2_args,
                compute_defines,
                kComputeKernelPath,
                /*fp32_dest_acc_en=*/true,
                create_unpack_to_dest_mode(),
                math_fidelity,
                /*math_approx_mode=*/false));
        }
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    // On a cache hit override_runtime_arguments() rewrites the buffer addresses (the kReader*Idx and k*BufferIdx slots)
    // on the reader and the writer. Row counts and offsets follow from the input shapes, which the program hash keys
    // on.
    if (use_balanced_parallelism) {
        assign_per_core_runtime_args_balanced(
            desc,
            grad_output_buffer,
            query_buffer,
            key_buffer,
            value_buffer,
            intermediates_buffer,
            u_scaler_buffer,
            grad_key_buffer,
            grad_value_buffer,
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
                tt::tt_metal::KernelDescriptor::RTArgList reader_args;
                reader_args.push_back(grad_output_buffer);
                reader_args.push_back(query_buffer);
                reader_args.push_back(key_buffer);
                reader_args.push_back(value_buffer);
                push_buffer_or_zero(reader_args, mask_buffer);
                reader_args.push_back(intermediates_buffer);
                reader_args.push_back(u_scaler_buffer);
                reader_args.push_back(num_rows);
                reader_args.push_back(start_row);
                desc.kernels[kReaderKernel].emplace_runtime_args(core, reader_args);
                tt::tt_metal::KernelDescriptor::RTArgList writer_args;
                writer_args.push_back(grad_key_buffer);
                writer_args.push_back(grad_value_buffer);
                writer_args.push_back(num_rows);
                writer_args.push_back(start_row);
                desc.kernels[kWriterKernel].emplace_runtime_args(core, writer_args);
                desc.kernels[in_group_1 ? kComputeGroup1Kernel : kComputeGroup2Kernel].emplace_runtime_args(
                    core, {start_row});
            });
    }

    return desc;
}

void SDPABackwardKVProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const kv::operation_attributes_t& operation_attributes,
    const kv::tensor_args_t& tensor_args,
    kv::tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    // Only buffer addresses change: row counts follow from the input shapes, which the program hash keys on.
    const auto* grad_output_buffer = tensor_args.grad_output.buffer();
    const auto* query_buffer = tensor_args.query.buffer();
    const auto* key_buffer = tensor_args.key.buffer();
    const auto* value_buffer = tensor_args.value.buffer();
    const auto* mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask.value().buffer() : nullptr;
    const auto* intermediates_buffer = tensor_args.intermediates.buffer();
    const auto* u_scaler_buffer = tensor_args.u_scaler.buffer();
    auto& [grad_key, grad_value] = tensor_return_value;
    const auto* grad_key_buffer = grad_key.buffer();
    const auto* grad_value_buffer = grad_value.buffer();

    // The work is K/V rows, as in create_descriptor(): one per K/V head and tile row of the sequence. Grouped-query
    // attention has fewer of them than query rows.
    const auto [qB, qNH, qS, qEmbd] = tensor_args.query.padded_shape().to_array_4D();
    const auto [kB, kNH, kS, kEmbd] = tensor_args.key.padded_shape().to_array_4D();
    const auto grid = tensor_args.query.device()->compute_with_storage_grid_size();
    const RowSplit row_split = choose_row_split(
        operation_attributes.mask_type,
        qS / tt::constants::TILE_HEIGHT,
        kB * kNH,
        static_cast<uint32_t>(grid.x * grid.y));

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernel);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernel);

    for_each_core(row_split.num_cores, static_cast<uint32_t>(grid.y), [&](const tt::tt_metal::CoreCoord& core) {
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kReaderGradOutputIdx] = grad_output_buffer->address();
            runtime_args[kReaderQueryIdx] = query_buffer->address();
            runtime_args[kReaderKeyIdx] = key_buffer->address();
            runtime_args[kReaderValueIdx] = value_buffer->address();
            runtime_args[kReaderMaskIdx] = mask_buffer != nullptr ? mask_buffer->address() : 0;
            runtime_args[kReaderIntermediatesIdx] = intermediates_buffer->address();
            runtime_args[kReaderUScalerIdx] = u_scaler_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kGradKeyBufferIdx] = grad_key_buffer->address();
            runtime_args[kGradValueBufferIdx] = grad_value_buffer->address();
        }
    });
}

}  // namespace ttml::metal::ops::sdpa_bw::device
