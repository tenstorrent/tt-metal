// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/dropout_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/qwen3_attention.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

// Qwen3 MLP - identical to Llama MLP but with explicit naming for clarity
class Qwen3MLP : public modules::ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_w1;  // gate projection
    std::shared_ptr<LinearLayer> m_w3;  // up projection
    std::shared_ptr<LinearLayer> m_w2;  // down projection
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    Qwen3MLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob = 0.0F);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

// Qwen3 Block - supports explicit head_dim specification with Q/K normalization
class Qwen3Block : public modules::ModuleBase {
private:
    std::shared_ptr<Qwen3MLP> m_mlp;
    std::shared_ptr<RMSNormLayer> m_input_layernorm;
    std::shared_ptr<RMSNormLayer> m_post_attention_layernorm;
    std::shared_ptr<Qwen3Attention> m_attention;  // Qwen3-specific attention with Q/K norms

public:
    // Constructor with explicit head_dim (for Qwen3)
    explicit Qwen3Block(
        uint32_t embedding_size,
        uint32_t num_heads,
        uint32_t num_groups,
        uint32_t head_dim,  // ← Explicit head dimension
        const ops::RotaryEmbeddingParams& rope_params,
        float dropout_prob = 0.0F,
        float rms_norm_eps = 1e-6F,  // Qwen3 uses 1e-6
        std::optional<uint32_t> intermediate_dim = std::nullopt);

    autograd::TensorPtr operator()(
        const autograd::TensorPtr& input,
        const autograd::TensorPtr& mask,
        std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
        const uint32_t layer_idx,
        const uint32_t new_tokens);

    // Getters for accessing sub-modules
    [[nodiscard]] std::shared_ptr<Qwen3Attention> get_attention() const {
        return m_attention;
    }
    [[nodiscard]] std::shared_ptr<RMSNormLayer> get_input_layernorm() const {
        return m_input_layernorm;
    }
    [[nodiscard]] std::shared_ptr<RMSNormLayer> get_post_attention_layernorm() const {
        return m_post_attention_layernorm;
    }
    [[nodiscard]] std::shared_ptr<Qwen3MLP> get_mlp() const {
        return m_mlp;
    }
};

}  // namespace ttml::modules
