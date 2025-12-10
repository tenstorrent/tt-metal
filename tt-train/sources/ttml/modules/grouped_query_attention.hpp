// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

enum class InferenceMode {
    PREFILL,  // Write entire prompt sequence into cache starting at position 0
    DECODE    // Write single new token into cache at cache_position
};

struct GQAConfig {
    uint32_t embedding_dim{};
    uint32_t num_heads{};
    uint32_t num_groups{};
    float dropout_prob{};
    std::reference_wrapper<const ops::RotaryEmbeddingParams> rope_params;
    bool bias_linears{false};
};

class GroupedQueryAttention : public ttml::modules::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_num_groups{};
    std::shared_ptr<ModuleBase> m_q_linear;
    std::shared_ptr<ModuleBase> m_kv_linear;
    std::shared_ptr<ModuleBase> m_out_linear;
    std::shared_ptr<ModuleBase> m_dropout;
    std::shared_ptr<RotaryEmbedding> m_embedding;

    // Helper functions for KV cache updates
    uint32_t update_cache_prefill(
        const tt::tt_metal::Tensor& key_tensor,
        const tt::tt_metal::Tensor& value_tensor,
        const autograd::TensorPtr& k_cache,
        const autograd::TensorPtr& v_cache);

    uint32_t update_cache_decode(
        const tt::tt_metal::Tensor& key_tensor,
        const tt::tt_metal::Tensor& value_tensor,
        const autograd::TensorPtr& k_cache,
        const autograd::TensorPtr& v_cache,
        const uint32_t cache_position);

public:
    explicit GroupedQueryAttention(const GQAConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Forward with KV cache for inference
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x,
        const autograd::TensorPtr& mask,
        const autograd::TensorPtr& k_cache,
        const autograd::TensorPtr& v_cache,
        const InferenceMode mode,
        const uint32_t cache_position);
};

}  // namespace ttml::modules
