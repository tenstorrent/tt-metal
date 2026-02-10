// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_block.hpp"

#include "modules/grouped_query_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

Qwen3MLP::Qwen3MLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob) {
    uint32_t multiple_of = 256;
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        const uint32_t unrounded_size = static_cast<uint32_t>(static_cast<float>(4 * embedding_size) * (2.0F / 3.0F));
        hidden_size = ((unrounded_size + multiple_of - 1U) / multiple_of) * multiple_of;
    }
    m_w1 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_w3 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_w2 = std::make_shared<LinearLayer>(hidden_size, embedding_size, /*has_bias=*/false);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("qwen3_mlp");
    register_module(m_w1, "w1");
    register_module(m_w3, "w3");
    register_module(m_w2, "w2");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr Qwen3MLP::operator()(const autograd::TensorPtr& input) {
    auto swished = ops::silu((*m_w1)(input));
    auto gate = (*m_w3)(input);
    auto gated = ops::mul(swished, gate);
    auto x = (*m_w2)(gated);
    x = (*m_dropout)(x);
    return x;
}

Qwen3Block::Qwen3Block(
    uint32_t embedding_size,
    uint32_t num_heads,
    uint32_t num_groups,
    uint32_t head_dim,
    const ops::RotaryEmbeddingParams& rope_params,
    float dropout_prob,
    float rms_norm_eps,
    std::optional<uint32_t> intermediate_dim) {
    m_mlp = std::make_shared<Qwen3MLP>(embedding_size, intermediate_dim, dropout_prob);
    // Qwen3 uses eps=1e-6 for all RMSNorm layers
    m_input_layernorm = std::make_shared<RMSNormLayer>(embedding_size, rms_norm_eps);
    m_post_attention_layernorm = std::make_shared<RMSNormLayer>(embedding_size, rms_norm_eps);

    // Qwen3 attention with Q/K normalization
    // Q: embedding_size → num_heads * head_dim
    // KV: embedding_size → num_groups * head_dim * 2
    // Q/K norms applied after projection but before RoPE (critical for stability!)
    // O: num_heads * head_dim → embedding_size
    m_attention = std::make_shared<Qwen3Attention>(Qwen3AttentionConfig{
        .embedding_dim = embedding_size,
        .num_heads = num_heads,
        .num_groups = num_groups,
        .head_dim = head_dim,
        .dropout_prob = dropout_prob,
        .rms_norm_eps = rms_norm_eps,
        .rope_params = rope_params,
        .bias_linears = false,
    });

    create_name("qwen3_block");
    register_module(m_mlp, "mlp");
    register_module(m_input_layernorm, "input_layernorm");
    register_module(m_post_attention_layernorm, "post_attention_layernorm");
    register_module(m_attention, "self_attn");
}

autograd::TensorPtr Qwen3Block::operator()(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& mask,
    std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
    const uint32_t layer_idx,
    const uint32_t new_tokens) {
    auto residual = input;
    auto h = (*m_input_layernorm)(input);
    h = (*m_attention)(h, mask, kv_cache, layer_idx, new_tokens);
    h = ops::add(h, residual);

    residual = h;
    auto x = (*m_post_attention_layernorm)(h);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);

    return x;
}

}  // namespace ttml::modules
