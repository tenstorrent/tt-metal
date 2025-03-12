// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scaled_dot_product_attention.hpp"

#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b, bool transpose_a, bool transpose_b) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
        /* core_grid */ ttnn::CoreGrid{7, 8},
        /* output_tile */ std::nullopt);
}

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
    const ttnn::Tensor& H_tensor, const ttnn::Tensor& G_tensor, bool transpose_a = false, bool transpose_b = false) {
    auto [B_H, H, S, E] = H_tensor.get_logical_shape().to_array_4D();
    auto [B_G, G, T, K] = G_tensor.get_logical_shape().to_array_4D();
    if (B_H != B_G) {
        throw std::invalid_argument("H_tensor and G_tensor must have the same batch size");
    }
    uint32_t B = B_H;
    if (H == G) {
        // no broadcasting needed
        return matmul(H_tensor, G_tensor, transpose_a, transpose_b);
    }
    // result will have shape (B, H, M, N)
    // we determine M,N based on the transpose options
    auto M = !transpose_a ? S : E;
    auto N = !transpose_b ? T : K;

    // - G != H:
    //   bcast G_tensor to groups in H_tensor then reshape back to H_tensor_shape:
    //   (B*G,H/G,M,E) x (B*G, 1, E, N) -> (B*G, H/G, M, N) -> (B, H, N, N)
    auto H_tensor_grouped = ttnn::reshape(H_tensor, ttnn::Shape{B * G, H / G, S, E});
    auto G_tensor_batched = ttnn::reshape(G_tensor, ttnn::Shape{B * G, 1U, T, K});

    // repeat G_tensor to group size for each group (manual bcast)
    ttnn::Tensor G_tensor_repeated = ttnn::repeat(G_tensor, ttnn::Shape{1U, H / G, 1U, 1U});
    auto bcasted_mm = matmul(H_tensor_grouped, G_tensor_repeated, transpose_a, transpose_b);
    auto reshaped_mm = ttnn::reshape(bcasted_mm, ttnn::Shape{B, H, M, N});
    return reshaped_mm;
}

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    if (!std::ranges::all_of(std::array{query, key, value}, [](const auto& t) { return t->get_rank() == 4U; })) {
        throw std::invalid_argument("query, key, and value must have rank 4");
    }

    auto [B, H, S, E] = query->get_value().get_logical_shape().to_array_4D();
    auto [BK, HK, SK, EK] = key->get_value().get_logical_shape().to_array_4D();
    auto [BV, HV, SV, EV] = value->get_value().get_logical_shape().to_array_4D();

    if (B != BK || B != BV || S != SK || S != SV || E != EK || E != EV) {
        throw std::invalid_argument("query, key, and value must have the same shape, except for the number of heads");
    }

    uint32_t G = H;            // number of KV groups, H for MHA mode
    uint32_t group_size = 1U;  // number of query heads per group, 1 for MHA mode
    if (H != HK || H != HV) {
        // grouped query mode
        if (HV != HK) {
            throw std::invalid_argument("query and key must have the same number of groups in grouped query mode");
        }
        G = HV;
        group_size = H / G;
        if (H % G != 0) {
            throw std::invalid_argument(
                "In grouped query mode, the number of heads must be divisible by the number of groups");
        }
    }

    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_logical_shape()[-1]));
    auto q_scaled = ttnn::experimental::mul(query->get_value(), scale);
    auto key_tensor = key->get_value();

    // σQ @ K
    ttnn::Tensor qk_scaled = group_shared_matmul(q_scaled, key_tensor, /*transpose_a=*/false, /*transpose_b=*/true);

    if (mask.has_value()) {
        auto mask_tensor = mask.value()->get_value();
        // ttnn::where when mask is not of the same shape as qk_scaled
        qk_scaled = ttnn::experimental::add(
            ttnn::experimental::mul(mask_tensor, qk_scaled),
            ttnn::experimental::mul(ttnn::experimental::sub(mask_tensor, 1.F), 1e9F));
    }
    // (B, H, S, S)
    auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    // TODO: add dropout here

    // softmax(σQ@K+mask) @ V
    ttnn::Tensor attention_qkv =
        group_shared_matmul(attention_weights, value->get_value(), /*transpose_a=*/false, /*transpose_b=*/false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad = [scale, query, key, value, attention_weights, out, mask, B, H, S, E, G]() {
        auto dL_dout = out->get_grad();  // (B, H, S, V)
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

        dL_dscaled_dot = ttnn::experimental::mul(dL_dscaled_dot, scale);  // [B,H,S,S]

        // dL_dQ = dL_dscaled_dot @ key
        ttnn::Tensor dL_dQ =
            group_shared_matmul(dL_dscaled_dot, key->get_value(), /*transpose_a=*/false, /*transpose_b=*/false);

        // helper function to collect grads from the query groups associated
        // with each key/value
        auto sum_over_groups = [&](const auto& ungrouped_grads) {
            if (G == H) {
                // group size is 1, nothing to do
                return ungrouped_grads;
            }
            // sum over groups:
            // [B,H,S,E] -> [B*G,H/G,S,E] -> [B*G,1,S,E] -> [B,G,S,E]
            auto grouped_grads = ttnn::reshape(ungrouped_grads, ttnn::Shape{B * G, H / G, S, E});
            auto summed_grads = ttnn_fixed::sum_moreh(grouped_grads, /*dim=*/1, /*keep_dim=*/true);
            return ttnn::reshape(summed_grads, ttnn::Shape{B, G, S, E});
        };

        // dL_dK = Σ_g [dL_dscaled_dot^T @ query]
        ttnn::Tensor dL_dK = matmul(
            dL_dscaled_dot,
            query->get_value(),
            /*transpose_a=*/true,
            /*transpose_b=*/false);
        dL_dK = sum_over_groups(dL_dK);  // no-op when G=H

        // dL_dV = Σ_g [attention_weights^T @ dL_dout]
        ttnn::Tensor dL_dV = matmul(
            attention_weights,
            dL_dout,
            /*transpose_a=*/true,
            /*transpose_b=*/false);
        dL_dV = sum_over_groups(dL_dV);  // no-op when G=H

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
    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_logical_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto qk_t = matmul(query->get_value(), key->get_value(), /* transpose_a */ false, /* transpose_b */ true);
    // (B, H, S, S) * scale
    auto qk_scaled = ttnn::multiply(qk_t, scale);
    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value()->get_value(), qk_scaled, /* other */ -1e9F);
    }
    // (B, H, S, S)
    // auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    auto attention_weights = ttnn::sigmoid(
        ttnn::subtract(qk_scaled, std::logf(static_cast<float>(query->get_value().get_logical_shape()[-2]))));

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv =
        matmul(attention_weights, value->get_value(), /* transpose_a */ false, /* transpose_b */ false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, qk_t, qk_scaled, attention_weights, attention_qkv, out, mask]() {
            auto grad_output = out->get_grad();
            // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
            auto grad_v = matmul(attention_weights, grad_output, /* transpose_a */ true, /* transpose_b */ false);
            auto grad_attention_weights =
                matmul(grad_output, value->get_value(), /* transpose_a */ false, /* transpose_b */ true);
            auto grad_scaled_dot =
                ttnn::sigmoid_bw(
                    grad_attention_weights,
                    ttnn::subtract(
                        qk_scaled, std::logf(static_cast<float>(query->get_value().get_logical_shape()[-2]))))
                    .front();

            if (mask.has_value()) {
                grad_scaled_dot = ttnn::where(mask.value()->get_value(), grad_scaled_dot, /* other */ 0.0F);
            }

            auto grad_q = matmul(
                grad_scaled_dot,
                key->get_value(),
                /* transpose_a */ false,
                /* transpose_b */ false);
            grad_q = ttnn::multiply(grad_q, scale);

            auto grad_k = matmul(
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
