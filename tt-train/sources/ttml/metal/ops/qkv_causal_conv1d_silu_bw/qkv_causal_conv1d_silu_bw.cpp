// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw.hpp"

#include "device/qkv_causal_conv1d_silu_bw_device_operation.hpp"

namespace ttml::metal {

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
    uint32_t channel_chunk_size) {
    TT_FATAL(
        input.storage_type() == ttnn::StorageType::DEVICE && input.logical_shape().rank() == 3,
        "qkv_causal_conv1d_silu_bw: input must be a [1,T,Q+K+V] device tensor");
    const ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams params{
        .sequence = static_cast<uint32_t>(input.logical_shape()[1]),
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .channel_chunk_size = channel_chunk_size,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
        .compute_kernel_config = ttnn::init_device_compute_kernel_config(
            input.device()->arch(),
            std::nullopt,
            tt::tt_metal::MathFidelity::HiFi4,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/true,
            /*default_l1_acc=*/false)};

    auto preact_outputs = ttnn::prim::ttml_qkv_causal_conv1d_silu_grad_preact(
        grad_q, grad_k, grad_v, input, history, tap0, tap1, tap2, tap3, params);
    auto grad_input =
        ttnn::prim::ttml_qkv_causal_conv1d_silu_grad_input(preact_outputs[0], tap0, tap1, tap2, tap3, params);

    const auto channels = static_cast<int32_t>(q_width + k_width + v_width);
    auto grad_taps = ttnn::sum(preact_outputs[1], 1, /*keepdim=*/true);
    const auto grad_tap = [&](int32_t tap, const ttnn::Tensor& like) {
        auto grad = ttnn::slice(
            grad_taps,
            ttsl::SmallVector<int32_t>{tap, 0, 0},
            ttsl::SmallVector<int32_t>{tap + 1, 1, channels},
            ttsl::SmallVector<int32_t>{1, 1, 1});
        return ttnn::reshape(grad, like.logical_shape());
    };
    return {grad_input, grad_tap(0, tap0), grad_tap(1, tap1), grad_tap(2, tap2), grad_tap(3, tap3)};
}

}  // namespace ttml::metal
