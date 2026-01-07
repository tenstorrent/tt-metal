// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"
#include "modules/rms_norm_module.hpp"
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
    std::shared_ptr<ModuleBase> m_q_linear;    // embedding_dim → num_heads * head_dim
    std::shared_ptr<ModuleBase> m_k_linear;    // embedding_dim → num_groups * head_dim
    std::shared_ptr<ModuleBase> m_v_linear;    // embedding_dim → num_groups * head_dim
    std::shared_ptr<ModuleBase> m_out_linear;  // num_heads * head_dim → embedding_dim
    std::shared_ptr<ModuleBase> m_dropout;
    std::shared_ptr<ModuleBase> m_embedding;  // RoPE
    std::shared_ptr<RMSNormLayer> m_q_norm;   // Q normalization per head (CRITICAL!)
    std::shared_ptr<RMSNormLayer> m_k_norm;   // K normalization per head (CRITICAL!)

public:
    explicit Qwen3Attention(const Qwen3AttentionConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Getters for accessing sub-modules
    [[nodiscard]] std::shared_ptr<ModuleBase> get_q_linear() const {
        return m_q_linear;
    }
    [[nodiscard]] std::shared_ptr<ModuleBase> get_k_linear() const {
        return m_k_linear;
    }
    [[nodiscard]] std::shared_ptr<ModuleBase> get_v_linear() const {
        return m_v_linear;
    }
    [[nodiscard]] std::shared_ptr<ModuleBase> get_out_linear() const {
        return m_out_linear;
    }
    [[nodiscard]] std::shared_ptr<RMSNormLayer> get_q_norm() const {
        return m_q_norm;
    }
    [[nodiscard]] std::shared_ptr<RMSNormLayer> get_k_norm() const {
        return m_k_norm;
    }
};

}  // namespace ttml::modules
