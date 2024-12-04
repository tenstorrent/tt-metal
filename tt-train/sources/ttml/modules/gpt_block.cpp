// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt_block.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

GPTMLP::GPTMLP(uint32_t embedding_size, float dropout_prob) {
    fc1 = std::make_shared<LinearLayer>(embedding_size, embedding_size * 4);
    fc2 = std::make_shared<LinearLayer>(embedding_size * 4, embedding_size);
    dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("gpt_mlp");
    register_module(fc1, "fc1");
    register_module(fc2, "fc2");
    register_module(dropout, "dropout");
}

autograd::TensorPtr GPTMLP::operator()(const autograd::TensorPtr& input) {
    auto x = (*fc1)(input);
    x = ops::gelu(x);
    x = (*fc2)(x);
    x = (*dropout)(x);
    return x;
}

GPTBlock::GPTBlock(uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm) {
    mlp = std::make_shared<GPTMLP>(embedding_size, dropout_prob);
    ln1 = std::make_shared<LayerNormLayer>(embedding_size, use_composite_layernorm);
    ln2 = std::make_shared<LayerNormLayer>(embedding_size, use_composite_layernorm);
    attention = std::make_shared<MultiHeadAttention>(embedding_size, num_heads, dropout_prob);

    create_name("gpt_block");
    register_module(mlp, "mlp");
    register_module(ln1, "ln1");
    register_module(ln2, "ln2");
    register_module(attention, "attention");
}

autograd::TensorPtr GPTBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    auto residual = input;
    auto x = (*ln1)(input);
    x = (*attention)(x, mask);
    x = ops::add(x, residual);

    residual = x;
    x = (*ln2)(x);
    x = (*mlp)(x);
    x = ops::add(x, residual);

    return x;
}

}  // namespace ttml::modules
