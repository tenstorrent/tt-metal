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
    using namespace ops::qkv_causal_conv1d_silu_bw::device;
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    constexpr uint32_t max_chunk = 256;
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    const uint64_t available_l1_bytes =
        device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    for (uint32_t chunk = std::min(channels, max_chunk); chunk >= tw; chunk -= tw) {
        if (channels % chunk == 0 && qkv_conv_bw_l1_bytes(chunk / tw, bf16, sizeof(uint16_t)) <= available_l1_bytes) {
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
    uint32_t sequence_splits) {
    TT_FATAL(
        input.storage_type() == ttnn::StorageType::DEVICE && input.logical_shape().rank() == 3,
        "qkv_causal_conv1d_silu_bw: input must be a [1,T,Q+K+V] device tensor");
    const auto sequence = static_cast<uint32_t>(input.logical_shape()[1]);
    const auto channels = q_width + k_width + v_width;
    if (channel_chunk_size == 0) {
        channel_chunk_size = qkv_causal_conv1d_silu_chunk_size(channels, *input.device());
    }
    if (sequence_splits == 0) {
        const auto grid = input.device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t blocks = std::max<uint32_t>(channels / std::max<uint32_t>(channel_chunk_size, 1), 1);
        const uint32_t Mt = std::max<uint32_t>(sequence / tt::constants::TILE_HEIGHT, 1);
        uint64_t best_cost = ~0ULL;
        for (uint32_t splits = 1; splits <= Mt; ++splits) {
            const uint64_t items = static_cast<uint64_t>(blocks) * splits;
            const uint64_t rounds = (items + cores - 1) / cores;
            const uint64_t tiles_per_item = (Mt + splits - 1) / splits + (splits > 1 ? 1 : 0);
            const uint64_t cost = rounds * tiles_per_item;
            if (cost < best_cost) {
                best_cost = cost;
                sequence_splits = splits;
            }
        }
    }
    const ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams params{
        .sequence = sequence,
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .channel_chunk_size = channel_chunk_size,
        .sequence_splits = sequence_splits,
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
    // Partials are fp32 [4, splits*32, C]: one fp32 row reduction (reducing an outer dim in fp32 is ~20x slower),
    // then cast the [4, 1, C] result once.
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
