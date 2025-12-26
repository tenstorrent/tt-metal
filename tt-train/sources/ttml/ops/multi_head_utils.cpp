// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_utils.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

#ifdef nlp_create_qkv_heads_program_factory_bug_fixed

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads) {
    // qkv shape is (B, 1, S, E * 3)
    // q, k, v shapes are (B, num_heads, S, E / num_heads)
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qkv->get_value(),
        std::nullopt,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, out_k, out_v, qkv]() {
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

    auto links_q = autograd::get_links(qkv);
    // grad_q function depends on gradients of q, k and v
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    // this needs to be added to make sure that gradients for k and v are computed before we run backward for q
    auto links_kv = autograd::get_links(qkv, out_q);
    out_k->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));
    out_v->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));
    return {out_q, out_k, out_v};
}

#else  // #ifdef nlp_create_qkv_heads_program_factory_bug_fixed

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads) {
    // qkv shape is (B, 1, S, E * 3)
    auto qkv_shape = qkv->get_value().logical_shape();
    auto batch_size = qkv_shape[0];
    auto seq_len = qkv_shape[2];
    auto total_dim = qkv_shape[3];
    auto embedding_dim = total_dim / 3;
    auto head_dim = embedding_dim / num_heads;

    // Manual head splitting to work around ttnn bug when head_dim < 32
    auto qkv_val = qkv->get_value();

    // Split QKV into Q, K, V along last dimension
    auto q_flat = ttnn::slice(
        qkv_val,
        ttnn::SmallVector<uint32_t>{0, 0, 0, 0},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, embedding_dim},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1});
    auto k_flat = ttnn::slice(
        qkv_val,
        ttnn::SmallVector<uint32_t>{0, 0, 0, embedding_dim},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, embedding_dim * 2},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1});
    auto v_flat = ttnn::slice(
        qkv_val,
        ttnn::SmallVector<uint32_t>{0, 0, 0, embedding_dim * 2},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, embedding_dim * 3},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1});

    // FIXED: Correct head splitting that preserves token-to-head mapping
    // Goal: [B, 1, S, E] -> [B, H, S, E/H] where each head gets contiguous dims from each token
    // Step 1: Remove channel dim: [B, 1, S, E] -> [B, S, E]
    auto q_no_channel = ttnn::reshape(q_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});
    // Step 2: Split embedding into heads: [B, S, E] -> [B, S, H, E/H]
    auto q_with_heads = ttnn::reshape(q_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});
    // Step 3: Transpose to put heads before sequence: [B, S, H, E/H] -> [B, H, S, E/H]
    auto q = ttnn::transpose(q_with_heads, 1, 2);

    auto k_no_channel = ttnn::reshape(k_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});
    auto k_with_heads = ttnn::reshape(k_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});
    auto k = ttnn::transpose(k_with_heads, 1, 2);

    auto v_no_channel = ttnn::reshape(v_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});
    auto v_with_heads = ttnn::reshape(v_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});
    auto v = ttnn::transpose(v_with_heads, 1, 2);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, out_k, out_v, qkv, batch_size, seq_len, embedding_dim]() {
        auto grad_q = out_q->get_grad();
        auto grad_k = out_k->get_grad();
        auto grad_v = out_v->get_grad();

        // Reverse the forward transformations: (B, H, S, E/H) -> (B, 1, S, E)
        // Step 1: Transpose back: [B, H, S, E/H] -> [B, S, H, E/H]
        grad_q = ttnn::transpose(grad_q, 1, 2);
        // Step 2: Merge heads: [B, S, H, E/H] -> [B, S, E]
        grad_q = ttnn::reshape(grad_q, ttnn::Shape{batch_size, seq_len, embedding_dim});
        // Step 3: Add channel dim back: [B, S, E] -> [B, 1, S, E]
        grad_q = ttnn::reshape(grad_q, ttnn::Shape{batch_size, 1, seq_len, embedding_dim});

        grad_k = ttnn::transpose(grad_k, 1, 2);
        grad_k = ttnn::reshape(grad_k, ttnn::Shape{batch_size, seq_len, embedding_dim});
        grad_k = ttnn::reshape(grad_k, ttnn::Shape{batch_size, 1, seq_len, embedding_dim});

        grad_v = ttnn::transpose(grad_v, 1, 2);
        grad_v = ttnn::reshape(grad_v, ttnn::Shape{batch_size, seq_len, embedding_dim});
        grad_v = ttnn::reshape(grad_v, ttnn::Shape{batch_size, 1, seq_len, embedding_dim});

        // Concatenate back to (B, 1, S, E*3)
        auto result = ttnn::concat(std::vector<ttnn::Tensor>({grad_q, grad_k, grad_v}), /* dim */ 3);
        qkv->add_grad(result);
    };

    auto links_q = autograd::get_links(qkv);
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    auto links_kv = autograd::get_links(qkv, out_q);
    out_k->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));
    out_v->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));

    return {out_q, out_k, out_v};
}

#endif  // #else // #ifdef nlp_create_qkv_heads_program_factory_bug_fixed

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x) {
    auto x_shape = x->get_value().logical_shape();

    uint32_t batch_size = x_shape[0];
    uint32_t num_heads = x_shape[1];
    uint32_t sequence_length = x_shape[2];
    uint32_t head_dim = x_shape[3];
    uint32_t embedding_dim = num_heads * head_dim;

    // FIXED: Manual head fusion to match the fixed heads_creation format
    // Goal: [B, H, S, E/H] -> [B, 1, S, E]
    // Step 1: Transpose to put sequence before heads: [B, H, S, E/H] -> [B, S, H, E/H]
    auto transposed = ttnn::transpose(x->get_value(), 1, 2);
    // Step 2: Merge heads into embedding: [B, S, H, E/H] -> [B, S, E]
    auto merged = ttnn::reshape(transposed, ttnn::Shape{batch_size, sequence_length, embedding_dim});
    // Step 3: Add channel dimension: [B, S, E] -> [B, 1, S, E]
    auto fused_heads = ttnn::reshape(merged, ttnn::Shape{batch_size, 1, sequence_length, embedding_dim});
    auto out = autograd::create_tensor(fused_heads);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim, head_dim]() {
        auto grad_output = out->get_grad();
        // Reverse the forward transformations: (B, 1, S, E) -> (B, H, S, E/H)
        // Step 1: Remove channel dim: [B, 1, S, E] -> [B, S, E]
        auto grad_no_channel = ttnn::reshape(grad_output, ttnn::Shape{batch_size, sequence_length, embedding_dim});
        // Step 2: Split embedding into heads: [B, S, E] -> [B, S, H, E/H]
        auto grad_with_heads =
            ttnn::reshape(grad_no_channel, ttnn::Shape{batch_size, sequence_length, num_heads, head_dim});
        // Step 3: Transpose to put heads before sequence: [B, S, H, E/H] -> [B, H, S, E/H]
        auto grad_result = ttnn::transpose(grad_with_heads, 1, 2);
        x->add_grad(grad_result);
    };

    auto links = autograd::get_links(x);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> grouped_heads_creation(
    const autograd::TensorPtr& qs, const autograd::TensorPtr& kvs, uint32_t num_heads, uint32_t num_groups) {
    // qs shape is (B, 1, S, E)
    // q shape is (B, num_heads, S, E/num_heads)
    // kvs shape is (B, 1, S, E*2)
    // k, v shapes are (B, num_groups, S, E / num_groups)
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

    auto links_q = autograd::get_links(qs, kvs);
    // grad_q function depends on gradients of q, k and v
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    // this needs to be added to make sure that gradients for k and v are computed before we run backward for q
    auto links_kv = autograd::get_links(qs, out_q);
    out_k->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));
    out_v->set_node(autograd::ctx().add_backward_node([]() {}, links_kv));
    return {out_q, out_k, out_v};
}

}  // namespace ttml::ops
