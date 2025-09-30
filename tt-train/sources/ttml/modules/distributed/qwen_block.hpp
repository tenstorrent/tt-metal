// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "autograd/module_base.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules::distributed {

class DistributedQwenMLP : public autograd::ModuleBase {
public:
    DistributedQwenMLP(
        uint32_t embedding_size, float dropout_prob, std::optional<uint32_t> intermediate_dim = std::nullopt);
    autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;

private:
    std::shared_ptr<autograd::ModuleBase> m_gate_proj;
    std::shared_ptr<autograd::ModuleBase> m_up_proj;
    std::shared_ptr<autograd::ModuleBase> m_down_proj;
    std::shared_ptr<autograd::ModuleBase> m_dropout;
};

class DistributedQwenBlock : public autograd::ModuleBase {
public:
    explicit DistributedQwenBlock(
        uint32_t embedding_size,
        uint32_t num_heads,
        uint32_t num_groups,
        const ops::RotaryEmbeddingParams& rope_params,
        float dropout_prob = 0.0F,
        std::optional<uint32_t> intermediate_dim = std::nullopt);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;

private:
    std::shared_ptr<autograd::ModuleBase> m_mlp;
    std::shared_ptr<autograd::ModuleBase> m_input_layernorm;
    std::shared_ptr<autograd::ModuleBase> m_post_attention_layernorm;
    std::shared_ptr<autograd::ModuleBase> m_self_attn;
};

}  // namespace ttml::modules::distributed
