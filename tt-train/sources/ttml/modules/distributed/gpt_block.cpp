// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt_block.hpp"

#include "modules/distributed/linear.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules::distributed {

DistributedGPTMLP::DistributedGPTMLP(uint32_t embedding_size, float dropout_prob) {
    fc1 = std::make_shared<ColumnParallelLinear>(
        embedding_size, embedding_size * 4, /* has_bias */ true, /* gather_output */ false);
    fc2 = std::make_shared<RowParallelLinear>(
        embedding_size * 4, embedding_size, /* has_bias */ true, /* input_is_parallel */ true);
    dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("gpt_mlp");
    register_module(fc1, "fc1");
    register_module(fc2, "fc2");
    register_module(dropout, "dropout");
}

autograd::TensorPtr DistributedGPTMLP::operator()(const autograd::TensorPtr& input) {
    auto x = (*fc1)(input);
    x = ops::gelu(x);
    x = (*fc2)(x);
    x = (*dropout)(x);
    return x;
}

DistributedGPTBlock::DistributedGPTBlock(
    uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm) {
    mlp = std::make_shared<DistributedGPTMLP>(embedding_size, dropout_prob);
    ln1 = std::make_shared<LayerNormLayer>(embedding_size, use_composite_layernorm);
    ln2 = std::make_shared<LayerNormLayer>(embedding_size, use_composite_layernorm);
    attention = std::make_shared<DistributedMultiHeadAttention>(embedding_size, num_heads, dropout_prob);

    create_name("gpt_block");
    register_module(mlp, "mlp");
    register_module(ln1, "ln1");
    register_module(ln2, "ln2");
    register_module(attention, "attention");
}

autograd::TensorPtr DistributedGPTBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
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

}  // namespace ttml::modules::distributed
