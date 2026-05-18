// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_block.hpp"

#include <fmt/core.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "grouped_query_attention.hpp"
#include "linear.hpp"
#include "modules/dropout_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/swiglu_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules::distributed {

DistributedLlamaMLP::DistributedLlamaMLP(
    uint32_t embedding_size, float dropout_prob, std::optional<uint32_t> intermediate_dim) {
    const auto& pctx = autograd::ctx().get_parallelism_context();
    auto tp_axis = pctx.get_tp_axis();
    bool use_tp = pctx.is_tp_enabled();

    uint32_t multiple_of = 256U;
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        const uint32_t unrounded_size = static_cast<uint32_t>(static_cast<float>(4U * embedding_size) * (2.0F / 3.0F));
        hidden_size = ((unrounded_size + multiple_of - 1U) / multiple_of) * multiple_of;
    }

    m_dropout_prob = dropout_prob;

    if (use_tp) {
        m_w1 = std::make_shared<ColumnParallelLinear>(
            embedding_size, hidden_size, /* has_bias */ false, /* gather_output */ false, tp_axis);
        m_w3 = std::make_shared<ColumnParallelLinear>(
            embedding_size, hidden_size, /* has_bias */ false, /* gather_output */ false, tp_axis);
        m_w2 = std::make_shared<RowParallelLinear>(
            hidden_size, embedding_size, /* has_bias */ false, /* input_is_parallel */ true, tp_axis);
    } else {
        m_w1_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_size, hidden_size, /* has_bias */ false);
        m_w3_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_size, hidden_size, /* has_bias */ false);
        m_w2_linear = std::make_shared<ttml::modules::LinearLayer>(hidden_size, embedding_size, /* has_bias */ false);
        m_w1 = m_w1_linear;
        m_w3 = m_w3_linear;
        m_w2 = m_w2_linear;
    }
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob, /* use_per_device_seed */ false);

    create_name("llama_mlp");
    register_module(m_w1, "w1");
    register_module(m_w3, "w3");
    register_module(m_w2, "w2");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr DistributedLlamaMLP::operator()(const autograd::TensorPtr& input) {
    const float dropout_prob = (get_run_mode() == RunMode::EVAL) ? 0.0F : m_dropout_prob;
    // Fused path is available only for non-TP where local LinearLayer weights exist.
    if (m_w1_linear && m_w2_linear && m_w3_linear) {
        // Keep distributed MLP RNG behavior aligned with DropoutLayer(use_per_device_seed=false).
        return ops::swiglu(
            input,
            m_w1_linear->get_weight(),
            m_w2_linear->get_weight(),
            m_w3_linear->get_weight(),
            dropout_prob,
            false);
    }

    auto linear_input = input;
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const bool use_sp = pctx.is_sp_enabled();
    if (use_sp) {
        static bool printed_sp_path = false;
        const int seq_dim = static_cast<int>(input->get_rank()) - 2;
        if (!printed_sp_path) {
            fmt::println(
                "[ttml][SP] LlamaMLP all_gathering sequence before column projections on seq_dim={} cluster_axis={}",
                seq_dim,
                pctx.get_tp_axis().has_value() ? static_cast<int>(*pctx.get_tp_axis()) : -1);
            printed_sp_path = true;
        }
        linear_input =
            ops::distributed::all_gather(input, seq_dim, pctx.get_tp_axis(), ops::distributed::GradOutputType::SHARDED);
    }

    auto swished = ops::silu(
        use_sp ? std::static_pointer_cast<ColumnParallelLinear>(m_w1)->forward_no_input_broadcast(linear_input)
               : (*m_w1)(linear_input));
    auto gate = use_sp ? std::static_pointer_cast<ColumnParallelLinear>(m_w3)->forward_no_input_broadcast(linear_input)
                       : (*m_w3)(linear_input);
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
    const autograd::TensorPtr& input, const std::optional<autograd::TensorPtr>& mask) {
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
