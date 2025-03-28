// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include "autograd/auto_context.hpp"
#include "linear.hpp"
#include "modules/dropout_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules::distributed {

DistributedGroupedQueryAttention::DistributedGroupedQueryAttention(const GQAConfig& config) :
    m_embedding_dim(config.embedding_dim), m_num_heads(config.num_heads), m_num_groups(config.num_groups) {
    auto num_devices = static_cast<uint32_t>(autograd::ctx().get_device().num_devices());
    if (m_num_heads % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Number of heads must be divisible by the number of devices. Number of heads = {}, devices = {}",
            m_num_heads,
            num_devices));
    }

    if (m_num_groups % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Number of groups must be divisible by the number of devices. Number of groups = {}, devices = {}",
            m_num_groups,
            num_devices));
    }

    m_num_local_heads = m_num_heads / num_devices;
    m_num_local_groups = m_num_groups / num_devices;

    // create layers
    m_q_linear = std::make_shared<ColumnParallelLinear>(
        m_embedding_dim, m_embedding_dim, /* has_bias */ false, /* gather_output */ false);
    auto concat_kv_dim = 2U * m_num_groups * (m_embedding_dim / m_num_heads);
    m_kv_linear = std::make_shared<ColumnParallelLinear>(
        m_embedding_dim, concat_kv_dim, /* has_bias */ false, /* gather_output */ false);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob, /* use_per_device_seed */ false);
    m_out_linear = std::make_shared<RowParallelLinear>(
        m_embedding_dim, m_embedding_dim, /* has_bias */ false, /* input_is_parallel */ true);
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
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_local_heads, m_num_local_groups);
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

}  // namespace ttml::modules::distributed
