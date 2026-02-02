// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/dropout_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/rms_norm_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

struct Qwen3AttentionConfig {
    uint32_t embedding_dim{};
    uint32_t num_heads{};
    uint32_t num_groups{};
    uint32_t head_dim{};  // Explicit head dimension (e.g., 128 for Qwen3-0.6B)
    float dropout_prob{};
    float rms_norm_eps{1e-6F};  // Qwen3 uses 1e-6 for Q/K norms
    std::reference_wrapper<const ops::RotaryEmbeddingParams> rope_params;
    bool bias_linears{false};
};

// Qwen3-specific Grouped Query Attention with Q/K normalization
// This is critical for numerical stability in Qwen3 models
// Uses separate Q, K, V projections (cleaner than combined KV)
class Qwen3Attention : public ttml::modules::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_num_groups{};
    uint32_t m_head_dim{};
    std::shared_ptr<LinearLayer> m_q_linear;
    std::shared_ptr<LinearLayer> m_k_linear;
    std::shared_ptr<LinearLayer> m_v_linear;
    std::shared_ptr<LinearLayer> m_out_linear;
    std::shared_ptr<DropoutLayer> m_dropout;
    std::shared_ptr<RotaryEmbedding> m_embedding;
    std::shared_ptr<RMSNormLayer> m_q_norm;
    std::shared_ptr<RMSNormLayer> m_k_norm;

public:
    explicit Qwen3Attention(const Qwen3AttentionConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x,
        const autograd::TensorPtr& mask,
        std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
        const uint32_t layer_idx,
        const uint32_t new_tokens);
};

}  // namespace ttml::modules
