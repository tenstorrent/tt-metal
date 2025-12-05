// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include <core/ttnn_all_includes.hpp>

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

uint32_t GroupedQueryAttention::update_cache_prefill(
    const tt::tt_metal::Tensor& key_tensor,
    const tt::tt_metal::Tensor& value_tensor,
    const ttml::autograd::TensorPtr& k_cache,
    const ttml::autograd::TensorPtr& v_cache) {
    tt::tt_metal::Tensor k_cache_tensor = k_cache->get_value();
    tt::tt_metal::Tensor v_cache_tensor = v_cache->get_value();
    auto cache_shape = k_cache_tensor.logical_shape();

    ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    uint32_t seq_len_to_process = key_tensor.logical_shape()[-2];

    ttnn::SmallVector<uint32_t> cache_start = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> cache_end = {cache_shape[0], cache_shape[1], seq_len_to_process, cache_shape[3]};

    ttnn::experimental::slice_write(key_tensor, k_cache_tensor, cache_start, cache_end, step);
    ttnn::experimental::slice_write(value_tensor, v_cache_tensor, cache_start, cache_end, step);

    k_cache->set_value(k_cache_tensor);
    v_cache->set_value(v_cache_tensor);

    return seq_len_to_process;
}

uint32_t GroupedQueryAttention::update_cache_decode(
    const tt::tt_metal::Tensor& key_tensor,
    const tt::tt_metal::Tensor& value_tensor,
    const ttml::autograd::TensorPtr& k_cache,
    const ttml::autograd::TensorPtr& v_cache,
    const uint32_t cache_position) {
    tt::tt_metal::Tensor k_cache_tensor = k_cache->get_value();
    tt::tt_metal::Tensor v_cache_tensor = v_cache->get_value();
    auto cache_shape = k_cache_tensor.logical_shape();
    auto kv_shape = key_tensor.logical_shape();

    ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> kv_end = {kv_shape[0], kv_shape[1], 1, kv_shape[3]};

    tt::tt_metal::Tensor single_key = ttnn::slice(key_tensor, token_start, kv_end, step);
    tt::tt_metal::Tensor single_value = ttnn::slice(value_tensor, token_start, kv_end, step);

    ttnn::SmallVector<uint32_t> cache_start = {0, 0, cache_position, 0};
    ttnn::SmallVector<uint32_t> cache_end = {cache_shape[0], cache_shape[1], cache_position + 1, cache_shape[3]};

    ttnn::experimental::slice_write(single_key, k_cache_tensor, cache_start, cache_end, step);
    ttnn::experimental::slice_write(single_value, v_cache_tensor, cache_start, cache_end, step);

    k_cache->set_value(k_cache_tensor);
    v_cache->set_value(v_cache_tensor);

    return cache_position + 1;
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    const ttml::autograd::TensorPtr& k_cache,
    const ttml::autograd::TensorPtr& v_cache,
    const InferenceMode mode,
    const uint32_t cache_position) {
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    uint32_t token_position = 0;
    if (mode == InferenceMode::DECODE) {
        token_position = cache_position;
    }

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads, token_position);
        key_with_heads = (*m_embedding)(key_with_heads, token_position);
    }

    const tt::tt_metal::Tensor& key_tensor = key_with_heads->get_value();
    const tt::tt_metal::Tensor& value_tensor = value_with_heads->get_value();
    auto kv_shape = key_tensor.logical_shape();

    constexpr uint32_t TILE_SIZE = 32;
    uint32_t seq_len_to_process = (mode == InferenceMode::PREFILL)
                                      ? update_cache_prefill(key_tensor, value_tensor, k_cache, v_cache)
                                      : update_cache_decode(key_tensor, value_tensor, k_cache, v_cache, cache_position);

    const uint32_t padded_seq_len = ((seq_len_to_process + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> token_end = {kv_shape[0], kv_shape[1], padded_seq_len, kv_shape[3]};

    const tt::tt_metal::Tensor& k_cache_slice = ttnn::slice(k_cache->get_value(), token_start, token_end, step);
    const tt::tt_metal::Tensor& v_cache_slice = ttnn::slice(v_cache->get_value(), token_start, token_end, step);

    auto k_cache_to_process = ttml::autograd::create_tensor(k_cache_slice);
    auto v_cache_to_process = ttml::autograd::create_tensor(v_cache_slice);

    auto attention =
        ttml::ops::scaled_dot_product_attention(query_with_heads, k_cache_to_process, v_cache_to_process, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
