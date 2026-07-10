// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "ops/multi_head_utils.hpp"

namespace ttml::modules {

MultiHeadAttention::MultiHeadAttention(uint32_t embedding_dim_, uint32_t num_heads_, float dropout_prob_) :
    m_embedding_dim(embedding_dim_), m_num_heads(num_heads_) {
    // create layers
    m_qkv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim * 3);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob_);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);

    // register modules
    create_name("multi_head_attention");
    register_module(m_qkv_linear, "qkv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
}

ttml::autograd::TensorPtr MultiHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const std::optional<ttml::autograd::TensorPtr>& mask) {
    auto qkv = (*m_qkv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] = ops::heads_creation(qkv, m_num_heads);

    // TODO(nuked-op sdpa): restore real call
    auto attention = query_with_heads;
    static_cast<void>(key_with_heads);
    static_cast<void>(value_with_heads);
    static_cast<void>(mask);

    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
