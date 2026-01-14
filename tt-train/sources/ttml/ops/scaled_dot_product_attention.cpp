// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scaled_dot_product_attention.hpp"

#include <cmath>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {
namespace {

// Wrapper around matmul to handle sharing of KV heads across groups of query
// heads.
// For e.g. Q @ V, there are two cases:
// - G == H: (B, H, S, S) x (B, H, S, V) -> (B, H, S, V)
// - G != H:
//    - In this case value has shape (B,G,S,V):
//      1. Reshape attention_weights to (B*G, H/G, S, S).
//      2. Reshape value to (B*G, 1, S, V).
//      3. Manually broadcast values over groupsize.
//      4. Matmul.
//      5. Reshape the result to (B, H, S, V).
//   - Summary of intermediate shapes:
//     (B*G, H/G, S, S) x (B*G, 1, S, V) -> (B*G, H/G, S, V) -> (B, H, S, V)
ttnn::Tensor group_shared_matmul(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& kv_tensor,
    bool transpose_a = false,
    bool transpose_b = false) {
    auto [batch_num, heads, seq_len, embedding_dim] = query_tensor.logical_shape().to_array_4D();
    auto [batch_num_v, groups, seq_len_v, embedding_dim_v] = kv_tensor.logical_shape().to_array_4D();
    if (batch_num != batch_num_v) {
        throw std::invalid_argument(fmt::format(
            "query_tensor and kv_tensor must have the same batch size, got shapes {} and {} respectively",
            query_tensor.logical_shape(),
            kv_tensor.logical_shape()));
    }
    if (heads == groups) {
        // no broadcasting needed
        return ttnn_fixed::matmul(query_tensor, kv_tensor, transpose_a, transpose_b);
    }
    // result will have shape (batch_num, heads, M, N)
    // we determine M,N based on the transpose options
    auto M = transpose_a ? embedding_dim : seq_len;
    auto N = transpose_b ? seq_len_v : embedding_dim_v;

    // - G != H:
    //   bcast kv_tensor to groups in query_tensor then reshape back to query_tensor_shape:
    //   (B*G,H/G,M,E) x (B*G, 1, E,N) -> (B*G, H/G, M, N) -> (B, H, M, N)
    auto query_tensor_grouped =
        ttnn::reshape(query_tensor, ttnn::Shape{batch_num * groups, heads / groups, seq_len, embedding_dim});
    auto kv_tensor_batched = ttnn::reshape(kv_tensor, ttnn::Shape{batch_num * groups, 1U, seq_len_v, embedding_dim_v});

    // repeat kv_tensor to group size for each group (manual bcast)
    ttnn::Tensor kv_tensor_repeated = ttnn::repeat(kv_tensor_batched, ttnn::Shape{1U, heads / groups, 1U, 1U});
    auto bcasted_mm = ttnn_fixed::matmul(query_tensor_grouped, kv_tensor_repeated, transpose_a, transpose_b);
    auto reshaped_mm = ttnn::reshape(bcasted_mm, ttnn::Shape{batch_num, heads, M, N});
    return reshaped_mm;
}

// helper function to collect grads from the query groups associated
// with each key/value
ttnn::Tensor sum_over_groups(const ttnn::Tensor& ungrouped_grads, uint32_t groups) {
    if (ungrouped_grads.logical_shape().rank() != 4) {
        throw std::invalid_argument(
            fmt::format("ungrouped_grads must have rank 4, but got rank {}", ungrouped_grads.logical_shape().rank()));
    }
    // [B,H,S,E]
    auto [batch_num, num_heads, seq_len, embedding_dim] = ungrouped_grads.logical_shape().to_array_4D();
    if (groups == num_heads) {
        // group size is 1, nothing to do
        return ungrouped_grads;
    }
    // sum over groups:
    // [B,H,S,E] -> [B*G,H/G,S,E] -> [B*G,1,S,E] -> [B,G,S,E]
    auto grouped_grads =
        ttnn::reshape(ungrouped_grads, ttnn::Shape{batch_num * groups, num_heads / groups, seq_len, embedding_dim});
    auto summed_grads = ttnn_fixed::sum_moreh(grouped_grads, /*dim=*/1, /*keep_dim=*/true);
    return ttnn::reshape(summed_grads, ttnn::Shape{batch_num, groups, seq_len, embedding_dim});
}

void validate_qkv_shapes(
    const autograd::TensorPtr& query, const autograd::TensorPtr& key, const autograd::TensorPtr& value) {
    if (!std::ranges::all_of(std::array{query, key, value}, [](const auto& t) { return t->get_rank() == 4U; })) {
        throw std::invalid_argument(fmt::format(
            "query, key, and value must have rank 4, but got ranks: query={}, key={}, value={}",
            query->get_rank(),
            key->get_rank(),
            value->get_rank()));
    }

    auto [batch_num, query_heads, seq_len, embedding_dim] = query->get_value().logical_shape().to_array_4D();
    auto [batch_num_key, key_heads, seq_len_key, embedding_dim_key] = key->get_value().logical_shape().to_array_4D();
    auto [batch_num_value, value_heads, seq_len_value, embedding_dim_value] =
        value->get_value().logical_shape().to_array_4D();

    if (batch_num != batch_num_key || batch_num != batch_num_value || seq_len_key != seq_len_value ||
        embedding_dim != embedding_dim_key || embedding_dim != embedding_dim_value) {
        throw std::invalid_argument(fmt::format(
            "Query, key, and value must have matching batch_num and embedding_dim. Key and value must have matching "
            "seq_len. Got shapes: query={}, key={}, value={}",
            query->get_value().logical_shape(),
            key->get_value().logical_shape(),
            value->get_value().logical_shape()));
    }

    uint32_t group_num = query_heads;  // (G) number of KV groups, H for MHA mode
    if (query_heads != key_heads || query_heads != value_heads) {
        // grouped query mode
        if (value_heads != key_heads) {
            throw std::invalid_argument(fmt::format(
                "query, key, and value must have the same number of groups in grouped query mode. Got: query heads={}, "
                "key heads={}, value heads={}",
                query_heads,
                key_heads,
                value_heads));
        }
        group_num = value_heads;
        if (query_heads % group_num != 0) {
            throw std::invalid_argument(fmt::format(
                "In grouped query mode, the number of query heads must be divisible by the number of key/value groups. "
                "Got: heads={}, groups={}",
                query_heads,
                group_num));
        }
    }
}

}  // namespace

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    validate_qkv_shapes(query, key, value);

    auto [batch_num, heads, seq_len, embedding_dim] = query->get_value().logical_shape().to_array_4D();
    auto groups = value->get_value().logical_shape().to_array_4D()[1];

    const float scale = 1.0F / std::sqrt(static_cast<float>(embedding_dim));
    constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};
    auto q_scaled =
        ttnn::multiply(query->get_value(), scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    auto key_tensor = key->get_value();

    // σQ @ K
    ttnn::Tensor qk_scaled = group_shared_matmul(q_scaled, key_tensor, /*transpose_a=*/false, /*transpose_b=*/true);

    if (mask.has_value()) {
        auto mask_tensor = mask.value()->get_value();
        // ttnn::where when mask is not of the same shape as qk_scaled
        qk_scaled = ttnn::add(
            ttnn::multiply(mask_tensor, qk_scaled, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            ttnn::multiply(
                ttnn::subtract(mask_tensor, 1.F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                1e9F,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
    }
    // (B, H, S, S)
    auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);
    // TODO: add dropout here

    // softmax(σQ@K+mask) @ V
    ttnn::Tensor attention_qkv =
        group_shared_matmul(attention_weights, value->get_value(), /*transpose_a=*/false, /*transpose_b=*/false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, attention_weights, out, mask, groups]() {
            constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};
            auto dL_dout = out->get_grad();  // (B, H, S, embedding_dim)
            // dL_d(softmax(σQK+mask)) = dL_dout @ value^T
            ttnn::Tensor dL_dattention_weights =
                group_shared_matmul(dL_dout, value->get_value(), /*transpose_a=*/false, /*transpose_b=*/true);

            auto dL_dscaled_dot = ttnn::moreh_softmax_backward(
                attention_weights,
                dL_dattention_weights,
                /* axis */ 3,
                /* output */ std::nullopt,
                ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMAX,
                ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
                /* output_mem_config */ std::nullopt,
                /* compute_kernel_config */ core::ComputeKernelConfig::precise());
            dL_dattention_weights.deallocate();

            dL_dscaled_dot = ttnn::multiply(
                dL_dscaled_dot, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // [B,H,S,S]

            // dL_dQ = dL_dscaled_dot @ key
            ttnn::Tensor dL_dQ =
                group_shared_matmul(dL_dscaled_dot, key->get_value(), /*transpose_a=*/false, /*transpose_b=*/false);

            // dL_dK = Σ_g [dL_dscaled_dot^T @ query]
            ttnn::Tensor dL_dK = ttnn_fixed::matmul(
                dL_dscaled_dot,
                query->get_value(),
                /*transpose_a=*/true,
                /*transpose_b=*/false);
            dL_dK = sum_over_groups(dL_dK, groups);  // no-op when groups == heads

            // dL_dV = Σ_g [attention_weights^T @ dL_dout]
            ttnn::Tensor dL_dV = ttnn_fixed::matmul(
                attention_weights,
                dL_dout,
                /*transpose_a=*/true,
                /*transpose_b=*/false);
            dL_dV = sum_over_groups(dL_dV, groups);  // no-op when groups == heads

            query->add_grad(dL_dQ);
            key->add_grad(dL_dK);
            value->add_grad(dL_dV);
        };

    auto links = autograd::get_links(query, key, value);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr scaled_dot_product_attention_fused(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask,
    float dropout_probability,
    bool fp32_dest_acc_en) {
    validate_qkv_shapes(query, key, value);

    // Get mask tensor if provided
    // Kernels support (1, 1, S, S) mask shape - same mask for all batches/heads
    std::optional<ttnn::Tensor> mask_tensor = std::nullopt;
    if (mask.has_value()) {
        mask_tensor = mask.value()->get_value();
    }

    // ========== Forward Pass using sdpa_fw kernel ==========
    auto fw_result = ttml::metal::sdpa_fw(
        query->get_value(),
        key->get_value(),
        value->get_value(),
        mask_tensor,
        dropout_probability,
        /*return_intermediates=*/true);  // Need intermediates for backward pass

    auto attn_output = fw_result[0].value();    // (B, H, S, D)
    auto intermediates = fw_result[1].value();  // (B, H, S, 2 tiles) - stores [max_val, 1/sum_exp] per row for softmax

    auto out = ttml::autograd::create_tensor(attn_output);

    // ========== Register Backward Function using sdpa_bw kernel ==========
    ttml::autograd::GradFunction grad =
        [query, key, value, mask_tensor, out, attn_output, intermediates, dropout_probability, fp32_dest_acc_en]() {
            auto grad_output = out->get_grad();

            // Call sdpa_bw kernel - returns [grad_Q, grad_K, grad_V]
            // dL_dQ: (B, H, S, D)
            // dL_dK: (B, G, S, D) for GQA, (B, H, S, D) for MHA
            // dL_dV: (B, G, S, D) for GQA, (B, H, S, D) for MHA
            auto [dL_dQ, dL_dK, dL_dV] = ttml::metal::sdpa_bw(
                grad_output,
                attn_output,
                query->get_value(),
                key->get_value(),
                value->get_value(),
                mask_tensor,
                intermediates,
                dropout_probability,
                fp32_dest_acc_en);

            query->add_grad(dL_dQ);
            key->add_grad(dL_dK);
            value->add_grad(dL_dV);
        };

    auto links = autograd::get_links(query, key, value);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr scaled_sigmoid_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    const float scale = 1.0F / std::sqrt(static_cast<float>(query->get_value().logical_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto qk_t =
        ttnn_fixed::matmul(query->get_value(), key->get_value(), /* transpose_a */ false, /* transpose_b */ true);
    // (B, H, S, S) * scale
    auto qk_scaled = ttnn::multiply(qk_t, scale);
    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value()->get_value(), qk_scaled, /* other */ -1e9F);
    }
    // (B, H, S, S)
    // auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    auto attention_weights =
        ttnn::sigmoid(ttnn::subtract(qk_scaled, std::log(static_cast<float>(query->get_value().logical_shape()[-2]))));

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv =
        ttnn_fixed::matmul(attention_weights, value->get_value(), /* transpose_a */ false, /* transpose_b */ false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, qk_t, qk_scaled, attention_weights, attention_qkv, out, mask]() {
            auto grad_output = out->get_grad();
            // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
            auto grad_v =
                ttnn_fixed::matmul(attention_weights, grad_output, /* transpose_a */ true, /* transpose_b */ false);
            auto grad_attention_weights =
                ttnn_fixed::matmul(grad_output, value->get_value(), /* transpose_a */ false, /* transpose_b */ true);
            auto grad_scaled_dot =
                ttnn::sigmoid_bw(
                    grad_attention_weights,
                    ttnn::subtract(qk_scaled, std::log(static_cast<float>(query->get_value().logical_shape()[-2]))))
                    .front();

            if (mask.has_value()) {
                grad_scaled_dot = ttnn::where(mask.value()->get_value(), grad_scaled_dot, /* other */ 0.0F);
            }

            auto grad_q = ttnn_fixed::matmul(
                grad_scaled_dot,
                key->get_value(),
                /* transpose_a */ false,
                /* transpose_b */ false);
            grad_q = ttnn::multiply(grad_q, scale);

            auto grad_k = ttnn_fixed::matmul(
                grad_scaled_dot,
                query->get_value(),
                /* transpose_a */ true,
                /* transpose_b */ false);
            grad_k = ttnn::multiply(grad_k, scale);

            query->add_grad(grad_q);
            key->add_grad(grad_k);
            value->add_grad(grad_v);
        };

    auto links = autograd::get_links(query, key, value);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
