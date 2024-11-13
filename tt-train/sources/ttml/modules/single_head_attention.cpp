// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "single_head_attention.hpp"

namespace ttml::modules {

SingleHeadAttention::SingleHeadAttention(uint32_t embedding_dim, float dropout_prob) {
    // create layers
    q_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    k_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    v_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob);
    out_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);

    // register modules
    create_name("single_head_attention");
    register_module(q_linear, "q_linear");
    register_module(k_linear, "k_linear");
    register_module(v_linear, "v_linear");
    register_module(dropout, "dropout");
    register_module(out_linear, "out_linear");
}

ttml::autograd::TensorPtr SingleHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto query = (*q_linear)(x);
    auto key = (*k_linear)(x);
    auto value = (*v_linear)(x);

    auto attention = ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    auto out = (*out_linear)(attention);
    out = (*dropout)(out);

    return out;
}

}  // namespace ttml::modules
