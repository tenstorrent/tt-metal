// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_block.hpp"

#include "modules/grouped_query_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {
LlamaMLP::LlamaMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob) {
    uint32_t multiple_of = 256;
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        const uint32_t unrounded_size = static_cast<uint32_t>(static_cast<float>(4 * embedding_size) * (2.0F / 3.0F));
        hidden_size = ((unrounded_size + multiple_of - 1U) / multiple_of) * multiple_of;
    }
    fmt::println("  BEFORE W1 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);
    m_w1 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    fmt::println("  AFTER W1 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);
    fmt::println("  BEFORE W3 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);
    m_w3 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    fmt::println("  AFTER W3 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);
    fmt::println("  BEFORE W2 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);
    m_w2 = std::make_shared<LinearLayer>(hidden_size, embedding_size, /*has_bias=*/false);
    fmt::println("  AFTER W2 CREATION EMB {} HIDDEN {}", embedding_size, hidden_size);

    fmt::println("  BEFORE DROPOUT CREATION WITH PROB {}", dropout_prob);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);
    fmt::println("  AFTER DROPOUT CREATION WITH PROB {}", dropout_prob);

    create_name("llama_mlp");
    register_module(m_w1, "w1");
    register_module(m_w3, "w3");
    register_module(m_w2, "w2");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr LlamaMLP::operator()(const autograd::TensorPtr& input) {
    auto swished = ops::silu((*m_w1)(input));
    auto gate = (*m_w3)(input);
    auto gated = ops::mul(swished, gate);
    auto x = (*m_w2)(gated);
    x = (*m_dropout)(x);
    return x;
}

LlamaBlock::LlamaBlock(
    uint32_t embedding_size,
    uint32_t num_heads,
    uint32_t num_groups,
    const ops::RotaryEmbeddingParams& rope_params,
    float dropout_prob,
    std::optional<uint32_t> intermediate_dim) {
    fmt::println("  BEFORE MLP CREATION");
    m_mlp = std::make_shared<LlamaMLP>(embedding_size, intermediate_dim, dropout_prob);
    fmt::println("  AFTER MLP CREATION");

    fmt::println("  BEFORE RMSNORM CREATION");
    m_attention_norm = std::make_shared<RMSNormLayer>(embedding_size);
    m_mlp_norm = std::make_shared<RMSNormLayer>(embedding_size);
    fmt::println("  AFTER RMSNORM CREATION");

    fmt::println("  BEFORE ATTENTION CREATION");
    m_attention = std::make_shared<GroupedQueryAttention>(GQAConfig{
        .embedding_dim = embedding_size,
        .num_heads = num_heads,
        .num_groups = num_groups,
        .dropout_prob = dropout_prob,
        .rope_params = rope_params,
        .bias_linears = false,
    });
    fmt::println("  AFTER ATTENTION CREATION");

    create_name("llama_block");
    register_module(m_mlp, "mlp");
    register_module(m_attention_norm, "attention_norm");
    register_module(m_mlp_norm, "mlp_norm");
    register_module(m_attention, "attention");
}

autograd::TensorPtr LlamaBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    auto residual = input;
    auto h = (*m_attention_norm)(input);
    h = (*m_attention)(h, mask);  // TODO: pass in start_pos, freqs_cis for RoPE here
    h = ops::add(h, residual);

    residual = h;
    auto x = (*m_mlp_norm)(h);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);
    ttml::autograd::ctx().get_profiler().read_results(&ttml::autograd::ctx().get_device(), "llama_block");

    return x;
}

}  // namespace ttml::modules
