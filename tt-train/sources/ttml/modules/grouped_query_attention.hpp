// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "modules/rotary_embedding.hpp"

namespace ttml::modules {

class GroupedQueryAttention : public ttml::autograd::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_num_groups{};
    std::shared_ptr<LinearLayer> m_q_linear;
    std::shared_ptr<LinearLayer> m_kv_linear;
    std::shared_ptr<LinearLayer> m_out_linear;
    std::shared_ptr<DropoutLayer> m_dropout;
    std::optional<std::shared_ptr<RotaryEmbedding>> m_rope;

public:
    explicit GroupedQueryAttention(
        uint32_t embedding_dim,
        uint32_t num_heads,
        uint32_t num_groups,
        float dropout_prob,
        const ops::RotaryEmbeddingParams* rope_params = nullptr);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
