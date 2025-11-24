// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_attention.hpp"

#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

Qwen3Attention::Qwen3Attention(const Qwen3AttentionConfig& config) :
    m_embedding_dim(config.embedding_dim),
    m_num_heads(config.num_heads),
    m_num_groups(config.num_groups),
    m_head_dim(config.head_dim) {
    // Separate Q, K, V projections (cleaner than combined KV!)
    uint32_t q_output_dim = m_num_heads * m_head_dim;
    uint32_t kv_output_dim = m_num_groups * m_head_dim;

    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, q_output_dim, config.bias_linears);
    m_k_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, kv_output_dim, config.bias_linears);
    m_v_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, kv_output_dim, config.bias_linears);

    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob);

    // Output projection: num_heads * head_dim → embedding_dim
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(q_output_dim, m_embedding_dim, config.bias_linears);

    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    m_q_norm = std::make_shared<RMSNormLayer>(m_head_dim, config.rms_norm_eps);
    m_k_norm = std::make_shared<RMSNormLayer>(m_head_dim, config.rms_norm_eps);

    // register modules
    create_name("qwen3_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_k_linear, "k_linear");
    register_module(m_v_linear, "v_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    register_module(m_embedding, "embedding");
    register_module(m_q_norm, "q_norm");
    register_module(m_k_norm, "k_norm");
}

ttml::autograd::TensorPtr Qwen3Attention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    // x shape: [B, 1, S, embedding_dim]

    // Separate Q, K, V projections
    auto q = (*m_q_linear)(x);  // [B, 1, S, num_heads * head_dim]
    auto k = (*m_k_linear)(x);  // [B, 1, S, num_groups * head_dim]
    auto v = (*m_v_linear)(x);  // [B, 1, S, num_groups * head_dim]

    auto q_shape = q->get_value().logical_shape();
    auto k_shape = k->get_value().logical_shape();
    auto B = q_shape[0];
    auto S = q_shape[2];

    auto q_reshaped = ttnn::reshape(q->get_value(), ttnn::Shape({B, 1, S * m_num_heads, m_head_dim}));
    auto k_reshaped = ttnn::reshape(k->get_value(), ttnn::Shape({B, 1, S * m_num_groups, m_head_dim}));

    q = (*m_q_norm)(autograd::create_tensor(q_reshaped));
    k = (*m_k_norm)(autograd::create_tensor(k_reshaped));

    q = autograd::create_tensor(ttnn::reshape(q->get_value(), q_shape));
    k = autograd::create_tensor(ttnn::reshape(k->get_value(), k_shape));

    // V is NOT normalized - use as-is!
    // Now we need to combine K and V for grouped_heads_creation
    auto kv = ttnn::concat(std::vector<ttnn::Tensor>({k->get_value(), v->get_value()}), /*dim=*/3);

    // Create heads from normalized Q and combined KV
    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, autograd::create_tensor(kv), m_num_heads, m_num_groups);

    // Apply RoPE AFTER normalization
    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);

    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
