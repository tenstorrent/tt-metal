// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_block.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

LlamaMLP::LlamaMLP(uint32_t embedding_size, float dropout_prob) {
    w1 = std::make_shared<LinearLayer>(embedding_size, embedding_size * 4);
    w3 = std::make_shared<LinearLayer>(embedding_size, embedding_size * 4);
    w2 = std::make_shared<LinearLayer>(embedding_size * 4, embedding_size);
    dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("llama_mlp");
    register_module(w1, "w1");
    register_module(w3, "w3");
    register_module(w2, "w2");
    register_module(dropout, "dropout");
}

autograd::TensorPtr LlamaMLP::operator()(const autograd::TensorPtr& input) {
    auto swished = ops::silu((*w1)(input));
    auto gate = (*w3)(input);
    auto gated = ops::mul(swished, gate);
    auto x = (*w2)(gated);
    x = (*dropout)(x);
    return x;
}

LlamaBlock::LlamaBlock(uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm) {
    mlp = std::make_shared<LlamaMLP>(embedding_size, dropout_prob);
    attention_norm = std::make_shared<RMSNormLayer>(embedding_size, use_composite_layernorm);
    ffn_norm = std::make_shared<RMSNormLayer>(embedding_size, use_composite_layernorm);
    attention = std::make_shared<MultiHeadAttention>(embedding_size, num_heads, dropout_prob);

    create_name("llama_block");
    register_module(mlp, "mlp");
    register_module(attention_norm, "attention_norm");
    register_module(ffn_norm, "ffn_norm");
    register_module(attention, "attention");
}

autograd::TensorPtr LlamaBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    auto residual = input;
    auto h = (*attention_norm)(input);
    h = (*attention)(h, mask);  // TODO: pass in start_pos, freqs_cis for RoPE here
    h = ops::add(h, residual);

    residual = h;
    auto x = (*ffn_norm)(h);
    x = (*mlp)(x);
    x = ops::add(x, residual);

    return x;
}

}  // namespace ttml::modules
