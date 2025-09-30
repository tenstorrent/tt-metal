// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen_block.hpp"

#include "modules/grouped_query_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

QwenMLP::QwenMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob) {
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        // Qwen2 uses 4 * embedding_size for intermediate dimension
        hidden_size = 4 * embedding_size;
    }

    // Qwen2 MLP structure: gate_proj, up_proj, down_proj
    m_gate_proj = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_up_proj = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_down_proj = std::make_shared<LinearLayer>(hidden_size, embedding_size, /*has_bias=*/false);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("qwen_mlp");
    register_module(m_gate_proj, "gate_proj");
    register_module(m_up_proj, "up_proj");
    register_module(m_down_proj, "down_proj");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr QwenMLP::operator()(const autograd::TensorPtr& input) {
    // Qwen2 MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
    auto gate = ops::silu((*m_gate_proj)(input));
    auto up = (*m_up_proj)(input);
    auto gated = ops::mul(gate, up);
    auto x = (*m_down_proj)(gated);
    x = (*m_dropout)(x);
    return x;
}

QwenBlock::QwenBlock(
    uint32_t embedding_size,
    uint32_t num_heads,
    uint32_t num_groups,
    const ops::RotaryEmbeddingParams& rope_params,
    float dropout_prob,
    std::optional<uint32_t> intermediate_dim) {
    m_mlp = std::make_shared<QwenMLP>(embedding_size, intermediate_dim, dropout_prob);
    m_input_layernorm = std::make_shared<RMSNormLayer>(embedding_size);
    m_post_attention_layernorm = std::make_shared<RMSNormLayer>(embedding_size);
    m_self_attn = std::make_shared<GroupedQueryAttention>(GQAConfig{
        .embedding_dim = embedding_size,
        .num_heads = num_heads,
        .num_groups = num_groups,
        .dropout_prob = dropout_prob,
        .rope_params = rope_params,
        .bias_linears = true,
    });

    create_name("qwen_block");
    register_module(m_mlp, "mlp");
    register_module(m_input_layernorm, "input_layernorm");
    register_module(m_post_attention_layernorm, "post_attention_layernorm");
    register_module(m_self_attn, "self_attn");
}

autograd::TensorPtr QwenBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    // Self Attention with residual connection
    auto residual = input;
    auto hidden_states = (*m_input_layernorm)(input);
    hidden_states = (*m_self_attn)(hidden_states, mask);
    hidden_states = ops::add(hidden_states, residual);

    // Feed Forward with residual connection
    residual = hidden_states;
    auto x = (*m_post_attention_layernorm)(hidden_states);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);

    ttml::autograd::ctx().get_profiler().read_results(&ttml::autograd::ctx().get_device(), "qwen_block");

    return x;
}

}  // namespace ttml::modules
