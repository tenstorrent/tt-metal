// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

GroupedQueryAttention::GroupedQueryAttention(
    uint32_t embedding_dim_,
    uint32_t num_heads_,
    uint32_t num_groups_,
    float dropout_prob_,
    const ops::RotaryEmbeddingParams* rope_params) :
    m_embedding_dim(embedding_dim_), m_num_heads(num_heads_), m_num_groups(num_groups_) {
    // create layers
    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);
    auto concat_kv_dim = 2 * m_num_groups * (m_embedding_dim / m_num_heads);
    m_kv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, concat_kv_dim);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob_);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);
    m_rope = std::nullopt;
    if (rope_params) {
        m_rope = std::make_shared<ttml::modules::RotaryEmbedding>(*rope_params);
    }

    // register modules
    create_name("grouped_query_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_kv_linear, "kv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    if (m_rope) {
        register_module(*m_rope, "rope");
    }
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);
    if (m_rope) {
        query_with_heads = (**m_rope)(query_with_heads);
        key_with_heads = (**m_rope)(key_with_heads);
    }

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);

    attention = ops::grouped_heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
