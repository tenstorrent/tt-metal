// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention.hpp"

#include "autograd/auto_context.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    auto& pctx = autograd::ctx().get_parallelism_context();
    std::optional<uint32_t> cp_axis = pctx.get_cp_axis();
    const uint32_t cp_size = pctx.get_cp_size();

    // If CP not enabled, fall back to regular SDPA
    if (!cp_axis.has_value() || cp_size <= 1) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const uint32_t cp_axis_value = cp_axis.value();

    auto k_current = key;
    auto v_current = value;

    auto [batch_num, heads, seq_len_local, dim] = query->get_value().logical_shape().to_array_4D();

    // Initialize accumulators for online softmax as TensorPtr for autograd
    // output_accum: weighted sum of outputs
    // global_scale: running total of unnormalized attention scales (sum_exp * exp_max)
    auto& device = autograd::ctx().get_device();
    autograd::TensorPtr output_accum = autograd::create_tensor(ttnn::zeros_like(query->get_value()));
    autograd::TensorPtr global_scale = autograd::create_tensor(ttnn::zeros(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(device)));

    // Create a ones tensor for reciprocal computation (1/x = ones/x)
    autograd::TensorPtr ones_scale = autograd::create_tensor(ttnn::ones(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(device)));

    // TODO: Implement mask support for ring attention
    // Mask handling requires per-device selection based on CP rank and ring step.
    // For now, mask is not supported in ring attention.
    // The mask parameter is accepted but ignored - full attention is computed.
    (void)mask;  // Suppress unused parameter warning

    for (uint32_t step = 0; step < cp_size; ++step) {
        // Get SDPA output with scale values for online softmax combination
        auto sdpa_result =
            ttml::ops::scaled_dot_product_attention_with_intermediates(query, k_current, v_current, std::nullopt);

        // Compute chunk scale = sum_exp * exp_max = sum(exp(qk_scaled))
        // This is the unnormalized attention weight sum for this KV chunk
        // Use autograd multiply operator for gradient propagation
        autograd::TensorPtr chunk_scale = sdpa_result.sum_exp * sdpa_result.exp_max;

        // New total scale
        autograd::TensorPtr new_global_scale = global_scale + chunk_scale;

        // Scale factors for weighted combination
        // old_weight = global_scale / new_global_scale
        // new_weight = chunk_scale / new_global_scale
        autograd::TensorPtr reciprocal_new_scale = ones_scale / new_global_scale;
        autograd::TensorPtr old_weight = global_scale * reciprocal_new_scale;
        autograd::TensorPtr new_weight = chunk_scale * reciprocal_new_scale;

        // Accumulate: output = old_output * old_weight + new_output * new_weight
        // Broadcasting is handled automatically by operator* (reduces gradients along broadcast dims)
        output_accum = output_accum * old_weight + sdpa_result.output * new_weight;
        global_scale = new_global_scale;

        // Ring shift KV to get next chunk (with autograd support)
        if (step < cp_size - 1) {
            k_current = ring_shift(k_current, cp_axis_value, true);
            v_current = ring_shift(v_current, cp_axis_value, true);
        }
    }

    return output_accum;
}

}  // namespace ttml::ops::distributed
