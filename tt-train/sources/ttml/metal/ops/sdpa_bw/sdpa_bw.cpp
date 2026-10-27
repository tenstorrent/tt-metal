// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw.hpp"

#include "device/sdpa_bw_kv_device_operation.hpp"
#include "device/sdpa_bw_q_device_operation.hpp"

namespace ttml::metal {

std::tuple<
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>>
sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    AttentionMaskType mask_type,
    const std::optional<ttnn::Tensor>& attn_mask,
    const float dropout_probability,
    const std::optional<ttnn::Tensor>& gate,
    const std::optional<ttnn::Tensor>& preallocated_grad_gate,
    const SDPABackwardGrads& grads) {
    const bool compute_grad_q = grads.query;
    const bool compute_grad_k = grads.key;
    const bool compute_grad_v = grads.value;
    const bool compute_grad_kv = compute_grad_k || compute_grad_v;
    // There is no grad_gate without a gate.
    const bool compute_grad_gate = grads.gate && gate.has_value();
    // Checked here rather than in the Q op, which is not launched for every request.
    TT_FATAL(
        !preallocated_grad_gate.has_value() || compute_grad_gate,
        "sdpa_bw: preallocated_grad_gate needs a gate and grads.gate");
    if (!compute_grad_q && !compute_grad_kv && !compute_grad_gate) {
        return {std::nullopt, std::nullopt, std::nullopt, std::nullopt};
    }

    // The Q kernel produces grad_Q, u_scaler = rowsum(dO * O) for the KV kernel's grad_K, and with a gate grad_gate
    // and the gated dO the KV kernel consumes in place of grad_output (for grad_K and grad_V alike). grad_V needs
    // neither u_scaler nor, without a gate, anything else from the Q kernel, so the Q kernel is skipped then.
    const bool compute_gated_grad_output = gate.has_value() && compute_grad_kv;
    std::optional<ttnn::Tensor> grad_Q;
    std::optional<ttnn::Tensor> u_scaler;
    std::optional<ttnn::Tensor> grad_gate;
    std::optional<ttnn::Tensor> gated_grad_output;
    if (compute_grad_q || compute_grad_k || compute_grad_gate || compute_gated_grad_output) {
        std::tie(grad_Q, u_scaler, grad_gate, gated_grad_output) = ttnn::prim::ttml_sdpa_q_bw(
            grad_output,
            attn_output,
            query,
            key,
            value,
            mask_type,
            attn_mask,
            intermediates,
            dropout_probability,
            /*preallocated_grad_query=*/std::nullopt,
            /*preallocated_u_scaler=*/std::nullopt,
            gate,
            preallocated_grad_gate,
            /*compute_grad_query=*/compute_grad_q,
            /*compute_u_scaler=*/compute_grad_k,
            /*compute_grad_gate=*/compute_grad_gate,
            /*compute_gated_grad_output=*/compute_gated_grad_output);
    }

    std::optional<ttnn::Tensor> grad_K;
    std::optional<ttnn::Tensor> grad_V;
    if (compute_grad_kv) {
        const auto& kv_grad_output = gated_grad_output.has_value() ? gated_grad_output.value() : grad_output;

        std::tie(grad_K, grad_V) = ttnn::prim::ttml_sdpa_kv_bw(
            kv_grad_output,
            query,
            key,
            value,
            mask_type,
            attn_mask,
            intermediates,
            u_scaler,
            dropout_probability,
            /*preallocated_grad_key=*/std::nullopt,
            /*preallocated_grad_value=*/std::nullopt,
            /*compute_grad_key=*/compute_grad_k,
            /*compute_grad_value=*/compute_grad_v);
    }

    // Each kernel only produces what was requested; the rest is nullopt.
    return {grad_Q, grad_K, grad_V, grad_gate};
}

}  // namespace ttml::metal
