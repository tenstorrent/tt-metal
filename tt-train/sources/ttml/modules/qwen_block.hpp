// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

class QwenMLP : public autograd::ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_gate_proj;
    std::shared_ptr<LinearLayer> m_up_proj;
    std::shared_ptr<LinearLayer> m_down_proj;
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    QwenMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob = 0.0F);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

class QwenBlock : public autograd::ModuleBase {
private:
    std::shared_ptr<QwenMLP> m_mlp;
    std::shared_ptr<RMSNormLayer> m_input_layernorm;
    std::shared_ptr<RMSNormLayer> m_post_attention_layernorm;
    std::shared_ptr<GroupedQueryAttention> m_self_attn;

public:
    explicit QwenBlock(
        uint32_t embedding_size,
        uint32_t num_heads,
        uint32_t num_groups,
        const ops::RotaryEmbeddingParams& rope_params,
        float dropout_prob = 0.0F,
        std::optional<uint32_t> intermediate_dim = std::nullopt);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules
