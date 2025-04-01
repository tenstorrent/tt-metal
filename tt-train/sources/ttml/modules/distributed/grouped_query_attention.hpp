// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules::distributed {

struct GQAConfig {
    uint32_t embedding_dim{};
    uint32_t num_heads{};
    uint32_t num_groups{};
    float dropout_prob{};
    std::reference_wrapper<const ops::RotaryEmbeddingParams> rope_params;
};

class DistributedGroupedQueryAttention : public ttml::autograd::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_num_local_heads{};
    uint32_t m_num_local_groups{};
    uint32_t m_num_groups{};
    std::shared_ptr<autograd::ModuleBase> m_q_linear;
    std::shared_ptr<autograd::ModuleBase> m_kv_linear;
    std::shared_ptr<autograd::ModuleBase> m_out_linear;
    std::shared_ptr<autograd::ModuleBase> m_dropout;
    std::shared_ptr<autograd::ModuleBase> m_embedding;

public:
    explicit DistributedGroupedQueryAttention(const GQAConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules::distributed
