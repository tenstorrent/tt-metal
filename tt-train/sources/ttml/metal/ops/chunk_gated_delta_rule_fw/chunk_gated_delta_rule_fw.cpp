// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_fw.hpp"

#include <cmath>

#include "device/chunk_gated_delta_rule_fw_device_operation.hpp"

namespace ttml::metal {

ChunkGatedDeltaRuleForwardResult chunk_gated_delta_rule_fw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t chunk_size,
    std::optional<float> scale,
    bool output_head_major,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using namespace ops::chunk_gated_delta_rule_fw::device;

    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const float scale_value = scale.has_value() ? *scale : 1.0F / std::sqrt(static_cast<float>(key_head_dim));
    const auto& mem = ttnn::DRAM_MEMORY_CONFIG;

    // The prep kernels take q already scaled, rounded to bf16 like ttnn.transformer.chunk_gated_delta_rule.
    const auto q_scaled = ttnn::multiply(q, scale_value);
    const auto prep = ttnn::prim::ttml_chunk_gated_delta_rule_fw_prep(
        q_scaled, k, v, g, beta, chunk_size, key_head_dim, kernel_config, mem);
    const uint32_t num_value_heads = g.logical_shape()[3];
    const auto scan = ttnn::prim::ttml_chunk_gated_delta_rule_fw_scan(
        prep, initial_state, num_value_heads, output_head_major, kernel_config, mem);

    return {
        .output = scan[kO],
        .final_state = scan[kFinalState],
        .h = scan[kH],
        .v_new = scan[kVNew],
        .g_cumsum = prep[kGCumsum],
        .a = prep[kA],
    };
}

}  // namespace ttml::metal
