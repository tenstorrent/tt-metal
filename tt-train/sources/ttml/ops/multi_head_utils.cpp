// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_utils.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads) {
    // qkv shape is (B, 1, S, E * 3)
    // q, k, v shapes should be (B, num_heads, S, E / num_heads)

    auto qkv_tensor = qkv->get_value();
    auto qkv_shape = qkv_tensor.logical_shape();

    uint32_t batch = qkv_shape[0];
    uint32_t seq_len = qkv_shape[2];
    uint32_t total_dim = qkv_shape[3];

    // WORKAROUND: The framework's nlp_create_qkv_heads has issues with concatenated QKV
    // Instead of trying to manually split (which requires complex slicing),
    // we'll use the function with separate Q and KV inputs

    // Expected: total_dim = 3 * embedding_dim
    uint32_t embedding_dim = total_dim / 3;

    // Use ttnn::slice with proper step parameter
    std::array<uint32_t, 4> step = {1, 1, 1, 1};

    // Extract Q: [:, :, :, 0:embedding_dim]
    std::array<uint32_t, 4> q_begin = {0, 0, 0, 0};
    std::array<uint32_t, 4> q_end = {batch, 1, seq_len, embedding_dim};
    auto q_concat = ttnn::slice(qkv_tensor, q_begin, q_end, step);

    // Extract K: [:, :, :, embedding_dim:2*embedding_dim]
    std::array<uint32_t, 4> k_begin = {0, 0, 0, embedding_dim};
    std::array<uint32_t, 4> k_end = {batch, 1, seq_len, 2 * embedding_dim};
    auto k_concat = ttnn::slice(qkv_tensor, k_begin, k_end, step);

    // Extract V: [:, :, :, 2*embedding_dim:3*embedding_dim]
    std::array<uint32_t, 4> v_begin = {0, 0, 0, 2 * embedding_dim};
    std::array<uint32_t, 4> v_end = {batch, 1, seq_len, 3 * embedding_dim};
    auto v_concat = ttnn::slice(qkv_tensor, v_begin, v_end, step);

    // Now use nlp_create_qkv_heads with separate Q and KV inputs
    auto kv_concat = ttnn::concat(std::vector<ttnn::Tensor>({k_concat, v_concat}), /* dim */ 3);

    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        q_concat,
        kv_concat,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, out_k, out_v, qkv]() {
        // Initialize gradients if not already initialized
        if (!out_q->is_grad_initialized()) {
            out_q->set_grad(core::zeros_like(out_q->get_value()));
        }
        if (!out_k->is_grad_initialized()) {
            out_k->set_grad(core::zeros_like(out_k->get_value()));
        }
        if (!out_v->is_grad_initialized()) {
            out_v->set_grad(core::zeros_like(out_v->get_value()));
        }

        auto grad_q = out_q->get_grad();
        auto grad_k = out_k->get_grad();
        auto grad_v = out_v->get_grad();
        // (B, num_heads, S, E / num_heads) -> (B, 1, S, E)
        grad_q = ttnn::experimental::nlp_concat_heads(grad_q);
        grad_k = ttnn::experimental::nlp_concat_heads(grad_k);
        grad_v = ttnn::experimental::nlp_concat_heads(grad_v);
        auto result = ttnn::concat(std::vector<ttnn::Tensor>({grad_q, grad_k, grad_v}), /* dim */ 3);
        qkv->add_grad(result);
    };

    // Add empty backward functions for k and v that initialize their gradients if needed
    autograd::GradFunction grad_k = [out_k]() {
        if (!out_k->is_grad_initialized()) {
            out_k->set_grad(core::zeros_like(out_k->get_value()));
        }
    };

    autograd::GradFunction grad_v = [out_v]() {
        if (!out_v->is_grad_initialized()) {
            out_v->set_grad(core::zeros_like(out_v->get_value()));
        }
    };

    auto links_q = autograd::get_links(qkv);
    // grad_q function depends on gradients of q, k and v
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    // this needs to be added to make sure that gradients for k and v are computed before we run backward for q
    auto links_kv = autograd::get_links(qkv, out_q);
    out_k->set_node(autograd::ctx().add_backward_node(std::move(grad_k), links_kv));
    out_v->set_node(autograd::ctx().add_backward_node(std::move(grad_v), links_kv));
    return {out_q, out_k, out_v};
}

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x) {
    auto x_shape = x->get_value().logical_shape();

    uint32_t batch_size = x_shape[0];
    uint32_t num_heads = x_shape[1];
    uint32_t sequence_length = x_shape[2];
    uint32_t embedding_dim = x_shape[3];

    // (B, H, S, E/H) -> (B, 1, S, E)
    auto fused_heads = ttnn::experimental::nlp_concat_heads(x->get_value());
    auto out = autograd::create_tensor(fused_heads);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim]() {
        auto grad_output = out->get_grad();
        // (B, 1, S, E) -> (B, 1, E, S)
        auto grad_result = ttnn::transpose(grad_output, -2, -1);
        // (B, 1, E, S) -> (B, H, E/H, S)
        grad_result = ttnn::reshape(grad_result, ttnn::Shape({batch_size, num_heads, embedding_dim, sequence_length}));
        // (B, H, E/H, S) -> (B, H, S, E/H)
        grad_result = ttnn::transpose(grad_result, -2, -1);
        x->add_grad(grad_result);
    };

    auto links = autograd::get_links(x);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> grouped_heads_creation(
    const autograd::TensorPtr& qs, const autograd::TensorPtr& kvs, uint32_t num_heads, uint32_t num_groups) {
    // WORKAROUND: The framework incorrectly enforces that Q and KV must have the same head dimension
    // For now, we'll work within the framework's limitations
    // This means GQA won't work correctly until the framework bug is fixed

    // The framework expects:
    // - Q tensor with shape [B, 1, S, E] where E will be divided by num_heads
    // - KV tensor with shape [B, 1, S, 2*E'] where E' will be divided by num_groups
    // But it incorrectly requires E/num_heads == E'/num_groups

    // For now, just call the function and let it fail with the assertion
    // This documents the bug clearly
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qs->get_value(),
        kvs->get_value(),
        /*num_q_heads=*/num_heads,
        /*num_kv_heads=*/num_groups,
        /*transpose_k_heads=*/false,
        /*memory_config=*/std::nullopt,
        /*optional_output_tensors=*/std::nullopt);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, out_k, out_v, qs, kvs]() {
        // Initialize gradients if not already initialized
        if (!out_q->is_grad_initialized()) {
            out_q->set_grad(core::zeros_like(out_q->get_value()));
        }
        if (!out_k->is_grad_initialized()) {
            out_k->set_grad(core::zeros_like(out_k->get_value()));
        }
        if (!out_v->is_grad_initialized()) {
            out_v->set_grad(core::zeros_like(out_v->get_value()));
        }

        auto grad_q = out_q->get_grad();
        auto grad_k = out_k->get_grad();
        auto grad_v = out_v->get_grad();
        // (B, num_heads, S, E / num_heads) -> (B, 1, S, E)
        grad_q = ttnn::experimental::nlp_concat_heads(grad_q);
        grad_k = ttnn::experimental::nlp_concat_heads(grad_k);
        grad_v = ttnn::experimental::nlp_concat_heads(grad_v);
        qs->add_grad(grad_q);
        auto kvs_grad = ttnn::concat(std::vector<ttnn::Tensor>({grad_k, grad_v}), /* dim */ 3);
        kvs->add_grad(kvs_grad);
    };

    // Add empty backward functions for k and v that initialize their gradients if needed
    autograd::GradFunction grad_k = [out_k]() {
        if (!out_k->is_grad_initialized()) {
            out_k->set_grad(core::zeros_like(out_k->get_value()));
        }
    };

    autograd::GradFunction grad_v = [out_v]() {
        if (!out_v->is_grad_initialized()) {
            out_v->set_grad(core::zeros_like(out_v->get_value()));
        }
    };

    auto links_q = autograd::get_links(qs, kvs);
    // grad_q function depends on gradients of q, k and v
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    // this needs to be added to make sure that gradients for k and v are computed before we run backward for q
    auto links_kv = autograd::get_links(qs, out_q);
    out_k->set_node(autograd::ctx().add_backward_node(std::move(grad_k), links_kv));
    out_v->set_node(autograd::ctx().add_backward_node(std::move(grad_v), links_kv));
    return {out_q, out_k, out_v};
}

}  // namespace ttml::ops
