// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw.hpp"

#include <algorithm>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>

#include "device/qkv_causal_conv1d_silu_bw_device_operation.hpp"

namespace ttml::metal {

uint32_t qkv_causal_conv1d_silu_chunk_size(uint32_t channels, const ttnn::distributed::MeshDevice& device) {
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    constexpr uint32_t max_chunk = 256;
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    const uint64_t available_l1_bytes =
        device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    for (uint32_t chunk = std::min(channels, max_chunk); chunk >= tw; chunk -= tw) {
        const uint64_t required_l1_bytes =
            ops::qkv_causal_conv1d_silu_bw::device::qkv_conv_bw_l1_bytes(chunk / tw, bf16, sizeof(uint16_t));
        if (channels % chunk == 0 && required_l1_bytes <= available_l1_bytes) {
            return chunk;
        }
    }
    return tw;
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
    if (channel_chunk_size == 0) {
        channel_chunk_size = qkv_causal_conv1d_silu_chunk_size(channels, *input.device());
    }
    if (row_splits == 0) {
        const auto grid = input.device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t channel_blocks = channels / channel_chunk_size;
        const uint32_t tile_rows = row_count / tt::constants::TILE_HEIGHT;
        // Cost of one work item in 32-row steps: its own steps, the halo step, and a fixed overhead for loading the
        // taps and filling / draining the reader-compute-writer pipeline. Without the overhead the model prefers a
        // second pass over the cores whenever it saves one step per item, and that loses on hardware: at
        // T=4096, chunk 256, 4 splits (1 pass) beat 9 (2 passes) by 2% at C=6144, and 2 splits beat 16 (7 passes)
        // by 12% at C=12288. Two steps of overhead reproduces both choices.
        constexpr uint64_t item_overhead_steps = 2;
        uint64_t best_cost = ~0ULL;
        for (uint32_t candidate = 1; candidate <= tile_rows; ++candidate) {
            const uint64_t items = static_cast<uint64_t>(channel_blocks) * candidate;
            const uint64_t rounds = (items + cores - 1) / cores;
            const uint64_t halo_steps = candidate > 1 ? 1 : 0;
            const uint64_t steps_per_item = (tile_rows + candidate - 1) / candidate + halo_steps + item_overhead_steps;
            const uint64_t cost = rounds * steps_per_item;
            if (cost < best_cost) {
                best_cost = cost;
                row_splits = candidate;
            }
        }
    }
    const ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams params{
        .row_count = row_count,
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .channel_chunk_size = channel_chunk_size,
        .row_splits = row_splits,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
        .compute_kernel_config = ttnn::init_device_compute_kernel_config(
            input.device()->arch(),
            std::nullopt,
            tt::tt_metal::MathFidelity::HiFi4,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/true,
            /*default_l1_acc=*/false)};

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
