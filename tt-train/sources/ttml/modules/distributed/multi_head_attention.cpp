// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "autograd/auto_context.hpp"
#include "modules/distributed/linear.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules::distributed {

DistributedMultiHeadAttention::DistributedMultiHeadAttention(
    uint32_t embedding_dim_, uint32_t num_heads_, float dropout_prob_) :
    m_embedding_dim(embedding_dim_), m_num_heads(num_heads_) {
    auto* device = &autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    if (m_num_heads % num_devices != 0) {
        throw std::runtime_error(fmt::format(
            "Number of heads must be divisible by the number of devices. Number of heads = {}, devices = {}",
            m_num_heads,
            num_devices));
    }
    m_local_num_heads = m_num_heads / num_devices;

    // create layers
    m_qkv_linear = std::make_shared<ColumnParallelLinear>(
        m_embedding_dim, m_embedding_dim * 3, /* has_bias */ true, /* gather_output */ false);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob_, /* use_per_device_seed */ false);
    m_out_linear = std::make_shared<RowParallelLinear>(
        m_embedding_dim, m_embedding_dim, /* has_bias */ true, /* input_is_parallel */ true);

    // register modules
    create_name("multi_head_attention");
    register_module(m_qkv_linear, "qkv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
}

ttml::autograd::TensorPtr DistributedMultiHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto qkv = (*m_qkv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] = ops::heads_creation(qkv, m_local_num_heads);

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);

    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules::distributed
