// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/grouped_query_attention.hpp"
#include "modules/linear_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

class LlamaMLP : public modules::ModuleBase {
public:
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr m_w1;
    ModuleBasePtr m_w3;
    ModuleBasePtr m_w2;
    ModuleBasePtr m_dropout;
    LlamaMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob = 0.0F);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

class LlamaBlock : public modules::ModuleBase {
public:
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr m_mlp;
    ModuleBasePtr m_attention_norm;
    ModuleBasePtr m_mlp_norm;
    ModuleBasePtr m_attention;
    explicit LlamaBlock(
        uint32_t embedding_size,
        uint32_t num_heads,
        uint32_t num_groups,
        const ops::RotaryEmbeddingParams& rope_params,
        float dropout_prob = 0.0F,
        std::optional<uint32_t> intermediate_dim = std::nullopt);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;

    // Forward with KV cache for inference
    autograd::TensorPtr operator()(
        const autograd::TensorPtr& input,
        const autograd::TensorPtr& mask,
        std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
        const uint32_t layer_idx,
        const uint32_t new_tokens);
};

}  // namespace ttml::modules
