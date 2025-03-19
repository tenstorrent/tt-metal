// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/distributed/multi_head_attention.hpp"
#include "modules/dropout_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"

namespace ttml::modules::distributed {

class DistributedGPTMLP : public autograd::ModuleBase {
public:
    DistributedGPTMLP(uint32_t embedding_size, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;

private:
    std::shared_ptr<distributed::ColumnParallelLinear> m_fc1;
    std::shared_ptr<distributed::RowParallelLinear> m_fc2;
    std::shared_ptr<DropoutLayer> m_dropout;
};

class DistributedGPTBlock : public autograd::ModuleBase {
public:
    explicit DistributedGPTBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm = false);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;

private:
    std::shared_ptr<DistributedGPTMLP> m_mlp;
    std::shared_ptr<LayerNormLayer> m_ln1;
    std::shared_ptr<LayerNormLayer> m_ln2;
    std::shared_ptr<DistributedMultiHeadAttention> m_attention;
};

}  // namespace ttml::modules::distributed
