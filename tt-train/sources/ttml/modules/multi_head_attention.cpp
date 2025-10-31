// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

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
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    // fmt::print("[MHA] Input shape: {}\n", x->get_value().logical_shape());
    // fmt::print("[MHA] embedding_dim: {}, num_heads: {}\n", m_embedding_dim, m_num_heads);

    auto qkv = (*m_qkv_linear)(x);
    // fmt::print("[MHA] After QKV linear: {}\n", qkv->get_value().logical_shape());

    auto [query_with_heads, key_with_heads, value_with_heads] = ops::heads_creation(qkv, m_num_heads);
    // fmt::print(
    //     "[MHA] Q shape: {}, K shape: {}, V shape: {}\n",
    //     query_with_heads->get_value().logical_shape(),
    //     key_with_heads->get_value().logical_shape(),
    //     value_with_heads->get_value().logical_shape());

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    // fmt::print("[MHA] After attention: {}\n", attention->get_value().logical_shape());

    attention = ops::heads_fusion(attention);
    // fmt::print("[MHA] After fusion: {}\n", attention->get_value().logical_shape());

    auto out = (*m_out_linear)(attention);
    // fmt::print("[MHA] After out_linear: {}\n", out->get_value().logical_shape());

    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
