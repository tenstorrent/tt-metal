// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_block.hpp"

#include "grouped_query_attention.hpp"
#include "linear.hpp"
#include "modules/dropout_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules::distributed {

DistributedLlamaMLP::DistributedLlamaMLP(
    uint32_t embedding_size, float dropout_prob, std::optional<uint32_t> intermediate_dim) {
    uint32_t multiple_of = 256U;
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        const uint32_t unrounded_size = static_cast<uint32_t>(static_cast<float>(4U * embedding_size) * (2.0F / 3.0F));
        hidden_size = ((unrounded_size + multiple_of - 1U) / multiple_of) * multiple_of;
    }
    m_w1 = std::make_shared<ColumnParallelLinear>(
        embedding_size, hidden_size, /* has_bias */ false, /* gather_output */ false);
    m_w3 = std::make_shared<ColumnParallelLinear>(
        embedding_size, hidden_size, /* has_bias */ false, /* gather_output */ false);
    m_w2 = std::make_shared<RowParallelLinear>(
        hidden_size, embedding_size, /* has_bias */ false, /* input_is_parallel */ true);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob, /* use_per_device_seed */ false);

    create_name("llama_mlp");
    register_module(m_w1, "w1");
    register_module(m_w3, "w3");
    register_module(m_w2, "w2");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr DistributedLlamaMLP::operator()(const autograd::TensorPtr& input) {
    auto swished = ops::silu((*m_w1)(input));
    auto gate = (*m_w3)(input);
    auto gated = swished * gate;
    auto x = (*m_w2)(gated);
    x = (*m_dropout)(x);
    return x;
}

DistributedLlamaBlock::DistributedLlamaBlock(
    uint32_t embedding_size,
    uint32_t num_heads,
    uint32_t num_groups,
    const ops::RotaryEmbeddingParams& rope_params,
    float dropout_prob,
    std::optional<uint32_t> intermediate_dim) {
    m_mlp = std::make_shared<DistributedLlamaMLP>(embedding_size, dropout_prob, intermediate_dim);
    m_attention_norm = std::make_shared<RMSNormLayer>(embedding_size);
    m_mlp_norm = std::make_shared<RMSNormLayer>(embedding_size);
    m_attention = std::make_shared<DistributedGroupedQueryAttention>(GQAConfig{
        .embedding_dim = embedding_size,
        .num_heads = num_heads,
        .num_groups = num_groups,
        .dropout_prob = dropout_prob,
        .rope_params = rope_params,
    });

    create_name("llama_block");
    register_module(m_mlp, "mlp");
    register_module(m_attention_norm, "attention_norm");
    register_module(m_mlp_norm, "mlp_norm");
    register_module(m_attention, "attention");
}

autograd::TensorPtr DistributedLlamaBlock::operator()(
    const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    auto residual = input;
    auto h = (*m_attention_norm)(input);
    h = (*m_attention)(h, mask);  // TODO: pass in start_pos, freqs_cis for RoPE here
    h = ops::add(h, residual);

    residual = h;
    auto x = (*m_mlp_norm)(h);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);

    return x;
}

}  // namespace ttml::modules::distributed
