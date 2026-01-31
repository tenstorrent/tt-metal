#include "qwen3_attention.hpp"

#include <array>
#include <core/ttnn_all_includes.hpp>

#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/concat_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/reshape_op.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

Qwen3Attention::Qwen3Attention(const Qwen3AttentionConfig& config) :
    m_embedding_dim(config.embedding_dim),
    m_num_heads(config.num_heads),
    m_num_groups(config.num_groups),
    m_head_dim(config.head_dim) {
    uint32_t q_output_dim = m_num_heads * m_head_dim;
    uint32_t kv_output_dim = m_num_groups * m_head_dim;

    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, q_output_dim, config.bias_linears);
    m_k_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, kv_output_dim, config.bias_linears);
    m_v_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, kv_output_dim, config.bias_linears);

    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob);

    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(q_output_dim, m_embedding_dim, config.bias_linears);

    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    m_q_norm = std::make_shared<RMSNormLayer>(m_head_dim, config.rms_norm_eps);
    m_k_norm = std::make_shared<RMSNormLayer>(m_head_dim, config.rms_norm_eps);

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
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
    const uint32_t layer_idx,
    const uint32_t new_tokens) {
    auto q = (*m_q_linear)(x);
    auto k = (*m_k_linear)(x);
    auto v = (*m_v_linear)(x);

    auto q_shape = q->get_value().logical_shape();
    auto k_shape = k->get_value().logical_shape();
    auto B = q_shape[0];
    auto S = q_shape[2];

    std::array<uint32_t, 4> q_reshaped_shape = {B, 1, S * m_num_heads, m_head_dim};
    std::array<uint32_t, 4> k_reshaped_shape = {B, 1, S * m_num_groups, m_head_dim};
    auto q_reshaped = ops::reshape(q, q_reshaped_shape);
    auto k_reshaped = ops::reshape(k, k_reshaped_shape);

    auto q_normed = (*m_q_norm)(q_reshaped);
    auto k_normed = (*m_k_norm)(k_reshaped);

    std::array<uint32_t, 4> q_shape_arr = {q_shape[0], q_shape[1], q_shape[2], q_shape[3]};
    std::array<uint32_t, 4> k_shape_arr = {k_shape[0], k_shape[1], k_shape[2], k_shape[3]};
    q = ops::reshape(q_normed, q_shape_arr);
    k = ops::reshape(k_normed, k_shape_arr);

    auto kv = ops::concat(std::vector<autograd::TensorPtr>{k, v}, 3);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    const uint32_t token_position = kv_cache ? kv_cache->get_cache_position() : 0;

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads, token_position);
        key_with_heads = (*m_embedding)(key_with_heads, token_position);
    }

    if (kv_cache) {
        std::tie(key_with_heads, value_with_heads) = models::common::transformer::update_kv_cache_and_get_slices(
            kv_cache, layer_idx, key_with_heads, value_with_heads, mask, new_tokens);
    }

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
