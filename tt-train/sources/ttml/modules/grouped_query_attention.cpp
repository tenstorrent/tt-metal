// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

GroupedQueryAttention::GroupedQueryAttention(const GQAConfig& config) :
    m_embedding_dim(config.embedding_dim), m_num_heads(config.num_heads), m_num_groups(config.num_groups) {
    // create layers
    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    auto concat_kv_dim = 2U * m_num_groups * (m_embedding_dim / m_num_heads);
    m_kv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, concat_kv_dim, config.bias_linears);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    // register modules
    create_name("grouped_query_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_kv_linear, "kv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    register_module(m_embedding, "embedding");
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    // Standard attention without KV cache
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
    const uint32_t layer_idx,
    const uint32_t new_tokens) {
    if (!kv_cache) {
        return operator()(x, mask);
    }

    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    // Get current cache position for RoPE
    const uint32_t token_position = kv_cache->get_cache_position();

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads, token_position);
        key_with_heads = (*m_embedding)(key_with_heads, token_position);
    }

    kv_cache->update(layer_idx, key_with_heads->get_value(), value_with_heads->get_value(), new_tokens);

    // Get cache tensors for attention computation
    const auto& k_cache = kv_cache->get_k_cache(layer_idx);
    const auto& v_cache = kv_cache->get_v_cache(layer_idx);

    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    const auto cache_shape = k_cache.logical_shape();
    // Use mask's key sequence length (last dimension) which is padded_whole_len
    // This matches the key sequence length expected by the attention operation
    const ttnn::SmallVector<uint32_t> token_end = {
        cache_shape[0], cache_shape[1], mask->get_value().logical_shape()[-1], cache_shape[3]};

    const tt::tt_metal::Tensor& k_cache_slice = ttnn::slice(k_cache, token_start, token_end, step);
    const tt::tt_metal::Tensor& v_cache_slice = ttnn::slice(v_cache, token_start, token_end, step);

    const auto k_cache_to_process = ttml::autograd::create_tensor(k_cache_slice);
    const auto v_cache_to_process = ttml::autograd::create_tensor(v_cache_slice);

    auto attention =
        ttml::ops::scaled_dot_product_attention(query_with_heads, k_cache_to_process, v_cache_to_process, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
