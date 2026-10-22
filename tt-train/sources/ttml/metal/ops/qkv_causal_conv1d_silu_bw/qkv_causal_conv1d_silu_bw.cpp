// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw.hpp"

#include <algorithm>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <vector>

#include "device/qkv_causal_conv1d_silu_bw_device_operation.hpp"

namespace ttml::metal {

namespace {

constexpr uint32_t max_chunk = 256;

uint64_t available_l1_bytes(const ttnn::distributed::MeshDevice& device) {
    return device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

bool chunk_fits_l1(uint32_t chunk, uint64_t l1_bytes) {
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    return ops::qkv_causal_conv1d_silu_bw::device::qkv_conv_bw_l1_bytes(
               chunk / tt::constants::TILE_WIDTH, bf16, sizeof(uint16_t)) <= l1_bytes;
}

}  // namespace

uint32_t qkv_causal_conv1d_silu_chunk_size(uint32_t channels, const ttnn::distributed::MeshDevice& device) {
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    const uint64_t l1_bytes = available_l1_bytes(device);
    for (uint32_t chunk = std::min(channels, max_chunk); chunk >= tw; chunk -= tw) {
        if (channels % chunk == 0 && chunk_fits_l1(chunk, l1_bytes)) {
            return chunk;
        }
    }
    return tw;
}

QkvCausalConv1dSiluBackwardWorkSplit qkv_causal_conv1d_silu_bw_work_split(
    uint32_t channels,
    uint32_t row_count,
    uint32_t cores,
    uint64_t l1_bytes,
    uint32_t channel_chunk_size,
    uint32_t row_splits) {
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    const uint32_t tile_rows = row_count / tt::constants::TILE_HEIGHT;
    // Cost of one (chunk, splits) choice, in tile units: passes over the core grid x (steps per item x cost of a
    // step + cost of an item). A step handles one 32-row tile row of channel_tiles tile columns and costs
    // channel_tiles + 2: the per-step NoC barriers and pipeline hand-offs do not shrink with the chunk. An item
    // does its own steps plus the halo step, and costs 4 * (channel_tiles + 1) on top: loading the taps, filling
    // and draining the reader-compute-writer pipeline, and writing the grad_tap partial. Fitted on a p300 with a
    // 110-core compute grid over chunks 32..256 and 1..32 splits at T=512 / 1024 / 4096 (7% mean error); it
    // picks the measured best at every shape: T=4096 C=6144 -> 256/4 (1 pass; 9 splits, 2 passes, is 2% slower),
    // T=4096 C=12288 -> 256/2 (16 splits, 7 passes, is 12% slower), T=512 C=6144 -> 128/2 and T=1024 C=3072 ->
    // 128/4, both 20% faster than 256 with any split. The pass boundaries move with the grid size.
    constexpr uint64_t step_overhead_tiles = 2;
    constexpr uint64_t item_overhead_tiles_per_tile = 4;
    const auto cost = [&](uint32_t chunk, uint32_t splits) {
        const uint64_t channel_tiles = chunk / tw;
        const uint64_t items = static_cast<uint64_t>(channels / chunk) * splits;
        const uint64_t rounds = (items + cores - 1) / cores;
        const uint64_t halo_steps = splits > 1 ? 1 : 0;
        const uint64_t steps_per_item = (tile_rows + splits - 1) / splits + halo_steps;
        return rounds * (steps_per_item * (channel_tiles + step_overhead_tiles) +
                         item_overhead_tiles_per_tile * (channel_tiles + 1));
    };

    std::vector<uint32_t> chunks;
    if (channel_chunk_size != 0) {
        chunks.push_back(channel_chunk_size);
    } else {
        // Chunk 32 makes the reader issue 64-byte row reads and runs 1.5x slower than chunk 64 at the same
        // split, more than the model predicts, so it is only a fallback.
        for (uint32_t chunk = std::min(channels, max_chunk); chunk >= 2 * tw; chunk -= tw) {
            if (channels % chunk == 0 && chunk_fits_l1(chunk, l1_bytes)) {
                chunks.push_back(chunk);
            }
        }
        if (chunks.empty()) {
            chunks.push_back(tw);
        }
    }
    const uint32_t first_splits = row_splits != 0 ? row_splits : 1;
    const uint32_t last_splits = row_splits != 0 ? row_splits : tile_rows;

    // Ties go to the wider chunk and the fewer splits: fewer, longer items, visited first.
    QkvCausalConv1dSiluBackwardWorkSplit best{chunks.front(), first_splits};
    uint64_t best_cost = ~0ULL;
    for (const uint32_t chunk : chunks) {
        for (uint32_t splits = first_splits; splits <= last_splits; ++splits) {
            const uint64_t candidate = cost(chunk, splits);
            if (candidate < best_cost) {
                best_cost = candidate;
                best = {chunk, splits};
            }
        }
    }
    return best;
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_bw(
    const ttnn::Tensor& grad_q,
    const ttnn::Tensor& grad_k,
    const ttnn::Tensor& grad_v,
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    uint32_t row_splits) {
    TT_FATAL(
        input.storage_type() == ttnn::StorageType::DEVICE && input.logical_shape().rank() == 3,
        "qkv_causal_conv1d_silu_bw: input must be a [1,T,Q+K+V] device tensor");
    const auto row_count = static_cast<uint32_t>(input.logical_shape()[1]);
    const auto channels = q_width + k_width + v_width;
    // One core gets channel_chunk_size channels x (row_count / row_splits) rows. Both default to auto.
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto work_split = qkv_causal_conv1d_silu_bw_work_split(
        channels, row_count, grid.x * grid.y, available_l1_bytes(*input.device()), channel_chunk_size, row_splits);
    const ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams params{
        .row_count = row_count,
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .channel_chunk_size = work_split.channel_chunk_size,
        .row_splits = work_split.row_splits,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG};

    auto outputs = ttnn::prim::ttml_qkv_causal_conv1d_silu_bw(
        grad_q, grad_k, grad_v, input, history, tap0, tap1, tap2, tap3, params);
    // Partials come out as [4, row_splits * 32, C] so one row reduction finishes them; see the writer's page order.
    auto grad_taps =
        ttnn::typecast(ttnn::sum(outputs[1], ttsl::SmallVector<int>{1}, /*keepdim=*/true), ttnn::DataType::BFLOAT16);
    const auto grad_tap = [&](int32_t tap, const ttnn::Tensor& like) {
        auto grad = ttnn::slice(
            grad_taps,
            ttsl::SmallVector<int32_t>{tap, 0, 0},
            ttsl::SmallVector<int32_t>{tap + 1, 1, static_cast<int32_t>(channels)},
            ttsl::SmallVector<int32_t>{1, 1, 1});
        return ttnn::reshape(grad, like.logical_shape());
    };
    return {outputs[0], grad_tap(0, tap0), grad_tap(1, tap1), grad_tap(2, tap2), grad_tap(3, tap3)};
}

}  // namespace ttml::metal
