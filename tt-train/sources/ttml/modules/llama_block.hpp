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

class LlamaMLP : public autograd::ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_w1;
    std::shared_ptr<LinearLayer> m_w3;
    std::shared_ptr<LinearLayer> m_w2;
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    LlamaMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob = 0.0F);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

class LlamaBlock : public autograd::ModuleBase {
private:
    std::shared_ptr<LlamaMLP> m_mlp;
    std::shared_ptr<RMSNormLayer> m_attention_norm;
    std::shared_ptr<RMSNormLayer> m_mlp_norm;
    std::shared_ptr<GroupedQueryAttention> m_attention;

public:
    explicit LlamaBlock(
        uint32_t embedding_size,
        uint32_t num_heads,
        uint32_t num_groups,
        const ops::RotaryEmbeddingParams& rope_params,
        float dropout_prob = 0.0F,
        std::optional<uint32_t> intermediate_dim = std::nullopt);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules
