// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include "autograd/auto_context.hpp"
#include "linear.hpp"
#include "modules/dropout_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/distributed/ring_attention.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules::distributed {

DistributedGroupedQueryAttention::DistributedGroupedQueryAttention(const GQAConfig& config) :
    m_embedding_dim(config.embedding_dim), m_num_heads(config.num_heads), m_num_groups(config.num_groups) {
    const auto& pctx = autograd::ctx().get_parallelism_context();
    auto tp_axis = pctx.get_tp_axis();
    auto tp_size = pctx.get_tp_size();

    if (m_num_heads % tp_size != 0) {
        throw std::runtime_error(fmt::format(
            "Number of heads must be divisible by the TP size. Number of heads = {}, TP size = {}",
            m_num_heads,
            tp_size));
    }

    if (m_num_groups % tp_size != 0) {
        throw std::runtime_error(fmt::format(
            "Number of groups must be divisible by the TP size. Number of groups = {}, TP size = {}",
            m_num_groups,
            tp_size));
    }

    m_num_local_heads = m_num_heads / tp_size;
    m_num_local_groups = m_num_groups / tp_size;

    // Calculate concat_kv_dim and ensure it's divisible by tp_size
    // concat_kv_dim = 2 * num_groups * head_dim, where head_dim = embedding_dim / num_heads
    auto head_dim = m_embedding_dim / m_num_heads;
    auto concat_kv_dim = 2U * m_num_groups * head_dim;

    TT_FATAL(
        concat_kv_dim % tp_size == 0 && m_embedding_dim % tp_size == 0,
        "KV concatenated dimension ({}) must be divisible by the TP size ({}). "
        "This requires: 2 * num_groups ({}) * head_dim ({}) % tp_size == 0",
        concat_kv_dim,
        tp_size,
        m_num_groups,
        head_dim);

    // create layers
    m_q_linear = std::make_shared<ColumnParallelLinear>(
        m_embedding_dim, m_embedding_dim, /* has_bias */ false, /* gather_output */ false, tp_axis);
    m_kv_linear = std::make_shared<ColumnParallelLinear>(
        m_embedding_dim, concat_kv_dim, /* has_bias */ false, /* gather_output */ false, tp_axis);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob, /* use_per_device_seed */ false);
    m_out_linear = std::make_shared<RowParallelLinear>(
        m_embedding_dim, m_embedding_dim, /* has_bias */ false, /* input_is_parallel */ true, tp_axis);
    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    // register modules
    create_name("grouped_query_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_kv_linear, "kv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    register_module(m_embedding, "embedding");
}

ttml::autograd::TensorPtr DistributedGroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x, const std::optional<ttml::autograd::TensorPtr>& mask) {
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_local_heads, m_num_local_groups);

    // Apply RoPE (build_rope_params auto-shards caches when CP is enabled)
    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    // Apply attention: use ring_attention_sdpa if CP is enabled, otherwise regular SDPA
    autograd::TensorPtr attention;
    auto& pctx = autograd::ctx().get_parallelism_context();
    if (pctx.is_cp_enabled() && pctx.get_cp_size() > 1) {
        /*
         * TODO: add support for non-causal mask
         */
        attention = ops::distributed::ring_attention_sdpa(
            query_with_heads, key_with_heads, value_with_heads, std::nullopt, ttml::metal::AttentionMaskType::Causal);
    } else {
        attention = ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    }

    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules::distributed
