// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bert_block.hpp"

#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

BertMLP::BertMLP(uint32_t embedding_dim, uint32_t intermediate_size, float dropout_prob) {
    m_dense = std::make_shared<LinearLayer>(embedding_dim, intermediate_size);
    m_output = std::make_shared<LinearLayer>(intermediate_size, embedding_dim);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("bert_mlp");
    register_module(m_dense, "dense");
    register_module(m_output, "output");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr BertMLP::operator()(const autograd::TensorPtr& input) {
    auto x = (*m_dense)(input);
    x = ops::gelu(x);  // BERT uses GELU activation
    x = (*m_output)(x);
    x = (*m_dropout)(x);
    return x;
}

BertAttention::BertAttention(uint32_t embedding_dim, uint32_t num_heads, float dropout_prob) {
    m_self_attention = std::make_shared<MultiHeadAttention>(embedding_dim, num_heads, dropout_prob);

    create_name("bert_attention");
    register_module(m_self_attention, "self_attention");
}

autograd::TensorPtr BertAttention::operator()(
    const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) {
    // MultiHeadAttention handles: QKV projection, attention, output projection, and dropout
    return (*m_self_attention)(input, attention_mask);
}

BertBlock::BertBlock(const BertBlockConfig& config) {
    m_attention = std::make_shared<BertAttention>(config.embedding_dim, config.num_heads, config.dropout_prob);

    // Disable hardware clamping to use BERT's exact epsilon (1e-12) instead of clamped value (1e-4)
    // BERT requires precise epsilon matching for accurate inference results
    m_attention_norm = std::make_shared<LayerNormLayer>(
        config.embedding_dim,
        config.layer_norm_eps,  // Pass BERT's epsilon (typically 1e-12)
        false,                  // use_composite_op = false
        false                   // enable_hardware_clamp = false (use exact epsilon)
    );

    m_mlp = std::make_shared<BertMLP>(config.embedding_dim, config.intermediate_size, config.dropout_prob);

    // Disable hardware clamping to use BERT's exact epsilon (1e-12) instead of clamped value (1e-4)
    // BERT requires precise epsilon matching for accurate inference results
    m_mlp_norm = std::make_shared<LayerNormLayer>(
        config.embedding_dim,
        config.layer_norm_eps,  // Pass BERT's epsilon (typically 1e-12)
        false,                  // use_composite_op = false
        false                   // enable_hardware_clamp = false (use exact epsilon)
    );

    create_name("bert_block");
    register_module(m_attention, "attention");
    register_module(m_attention_norm, "attention_norm");
    register_module(m_mlp, "mlp");
    register_module(m_mlp_norm, "mlp_norm");
}

autograd::TensorPtr BertBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) {
    // Self-attention with residual connection and layer norm
    // BERT uses post-norm: LayerNorm(x + Attention(x))
    auto attention_output = (*m_attention)(input, attention_mask);
    auto attention_residual = ops::add(attention_output, input);
    attention_residual = (*m_attention_norm)(attention_residual);

    // Feed-forward with residual connection and layer norm
    auto mlp_output = (*m_mlp)(attention_residual);
    auto mlp_residual = ops::add(mlp_output, attention_residual);
    mlp_residual = (*m_mlp_norm)(mlp_residual);

    return mlp_residual;
}

BertBlock::IntermediateOutputs BertBlock::forward_with_intermediates(
    const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) {
    IntermediateOutputs outputs;

    // Self-attention with residual connection and layer norm
    auto attention_output = (*m_attention)(input, attention_mask);
    auto attention_residual = ops::add(attention_output, input);
    attention_residual = (*m_attention_norm)(attention_residual);
    outputs.attention_output = attention_residual;

    // Feed-forward with residual connection and layer norm
    auto mlp_output = (*m_mlp)(attention_residual);
    auto mlp_residual = ops::add(mlp_output, attention_residual);
    mlp_residual = (*m_mlp_norm)(mlp_residual);
    outputs.block_output = mlp_residual;

    return outputs;
}

}  // namespace ttml::modules
