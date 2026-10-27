// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_bw.hpp"

#include <cmath>

#include "core/compute_kernel_config.hpp"
#include "device/chunk_gated_delta_rule_bw_device_operation.hpp"

namespace ttml::metal {

namespace {

// [G*B, 1, T, X] planes -> [B, 1, T, X], summing the G copies each key head received.
ttnn::Tensor sum_value_head_planes(const ttnn::Tensor& planes, uint32_t G) {
    if (G == 1U) {
        return planes;
    }
    const auto& s = planes.logical_shape();
    const uint32_t B = s[0] / G;
    auto grouped = ttnn::reshape(planes, ttnn::Shape({G, B, s[2], s[3]}));
    auto summed = ttnn::sum(
        grouped,
        /* dim_arg */ ttsl::SmallVector<int>{0},
        /* keep_dim */ true,
        /* output_mem_config */ planes.memory_config(),
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    return ttnn::reshape(summed, ttnn::Shape({B, 1U, s[2], s[3]}));
}

}  // namespace

std::vector<std::optional<ttnn::Tensor>> chunk_gated_delta_rule_bw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& d_out,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& d_final_state,
    const std::optional<ttnn::Tensor>& h,
    const std::optional<ttnn::Tensor>& v_new,
    const std::optional<ttnn::Tensor>& a,
    uint32_t chunk_size,
    std::optional<float> scale,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using namespace ops::chunk_gated_delta_rule_bw::device;

    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const float scale_value = scale.has_value() ? *scale : 1.0F / std::sqrt(static_cast<float>(key_head_dim));

    auto outputs = ttnn::prim::ttml_chunk_gated_delta_rule_bw(
        q,
        k,
        v,
        g,
        beta,
        d_out,
        initial_state,
        d_final_state,
        h,
        v_new,
        a,
        chunk_size,
        key_head_dim,
        scale_value,
        kernel_config,
        memory_config.value_or(q.memory_config()));

    const uint32_t G = outputs[kDq].logical_shape()[0] / q.logical_shape()[0];
    std::optional<ttnn::Tensor> dh0;
    if (initial_state.has_value()) {
        dh0 = outputs[kDh0];
    }
    const auto like_q = [&](const ttnn::Tensor& planes) {
        auto summed = sum_value_head_planes(planes, G);
        return q.logical_shape().rank() == 4U ? summed : ttnn::reshape(summed, q.logical_shape());
    };
    return {
        like_q(outputs[kDq]),
        like_q(outputs[kDk]),
        outputs[kDv],
        outputs[kDg],
        outputs[kDbeta],
        dh0,
    };
}

}  // namespace ttml::metal
