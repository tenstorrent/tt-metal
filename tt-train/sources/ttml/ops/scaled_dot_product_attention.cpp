// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scaled_dot_product_attention.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_logical_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto q_scaled = ttnn::multiply(query->get_value(), scale);
    auto qk_scaled = ttnn_fixed::matmul(
        q_scaled,
        key->get_value(),
        /* transpose_a */ false,
        /* transpose_b */ true,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());

    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value()->get_value(), qk_scaled, /* other */ -1e9F);
    }
    // (B, H, S, S)
    auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    // TODO: add dropout here

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv = ttnn_fixed::matmul(
        attention_weights,
        value->get_value(),
        /* transpose_a */ false,
        /* transpose_b */ false,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad = [scale, query, key, value, attention_weights, out, mask]() {
        auto grad_output = out->get_grad();
        // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
        auto grad_v = ttnn_fixed::matmul(
            attention_weights,
            grad_output,
            /* transpose_a */ true,
            /* transpose_b */ false,
            /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
        auto grad_attention_weights = ttnn_fixed::matmul(
            grad_output,
            value->get_value(),
            /* transpose_a */ false,
            /* transpose_b */ true,
            /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
        auto grad_scaled_dot = ttnn::moreh_softmax_backward(
            attention_weights,
            grad_attention_weights,
            /* axis */ 3,
            /* output */ std::nullopt,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMAX,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());

        grad_scaled_dot = ttnn::multiply(grad_scaled_dot, scale);
        auto grad_q = ttnn_fixed::matmul(
            grad_scaled_dot,
            key->get_value(),
            /* transpose_a */ false,
            /* transpose_b */ false,
            /* compute_kernel_config */ core::ComputeKernelConfig::matmul());

        auto grad_k = ttnn_fixed::matmul(
            grad_scaled_dot,
            query->get_value(),
            /* transpose_a */ true,
            /* transpose_b */ false,
            /* compute_kernel_config */ core::ComputeKernelConfig::matmul());

        query->add_grad(grad_q);
        key->add_grad(grad_k);
        value->add_grad(grad_v);
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
    auto qk_t = ttnn_fixed::matmul(
        query->get_value(),
        key->get_value(),
        /* transpose_a */ false,
        /* transpose_b */ true,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
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
    auto attention_qkv = ttnn_fixed::matmul(
        attention_weights,
        value->get_value(),
        /* transpose_a */ false,
        /* transpose_b */ false,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, qk_t, qk_scaled, attention_weights, attention_qkv, out, mask]() {
            auto grad_output = out->get_grad();
            // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
            auto grad_v = ttnn_fixed::matmul(
                attention_weights,
                grad_output,
                /* transpose_a */ true,
                /* transpose_b */ false,
                /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
            auto grad_attention_weights = ttnn_fixed::matmul(
                grad_output,
                value->get_value(),
                /* transpose_a */ false,
                /* transpose_b */ true,
                /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
            auto grad_scaled_dot =
                ttnn::sigmoid_bw(
                    grad_attention_weights,
                    ttnn::subtract(
                        qk_scaled, std::logf(static_cast<float>(query->get_value().get_logical_shape()[-2]))))
                    .front();

            if (mask.has_value()) {
                grad_scaled_dot = ttnn::where(mask.value()->get_value(), grad_scaled_dot, /* other */ 0.0F);
            }

            auto grad_q = ttnn_fixed::matmul(
                grad_scaled_dot,
                key->get_value(),
                /* transpose_a */ false,
                /* transpose_b */ false,
                /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
            grad_q = ttnn::multiply(grad_q, scale);

            auto grad_k = ttnn_fixed::matmul(
                grad_scaled_dot,
                query->get_value(),
                /* transpose_a */ true,
                /* transpose_b */ false,
                /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
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
