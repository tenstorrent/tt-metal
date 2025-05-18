// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

struct GQAConfig {
    uint32_t embedding_dim{};
    uint32_t num_heads{};
    uint32_t num_groups{};
    float dropout_prob{};
    std::reference_wrapper<const ops::RotaryEmbeddingParams> rope_params;
    bool bias_linears{false};
};

class GroupedQueryAttention : public ttml::autograd::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_num_groups{};
    std::shared_ptr<ModuleBase> m_q_linear;
    std::shared_ptr<ModuleBase> m_kv_linear;
    std::shared_ptr<ModuleBase> m_out_linear;
    std::shared_ptr<ModuleBase> m_dropout;
    std::shared_ptr<ModuleBase> m_embedding;

public:
    explicit GroupedQueryAttention(const GQAConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
