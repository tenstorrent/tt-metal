// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/distributed/multi_head_attention.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"

namespace ttml::modules::distributed {

class DistributedGPTMLP : public autograd::ModuleBase {
    std::shared_ptr<distributed::ColumnParallelLinear> fc1;
    std::shared_ptr<distributed::RowParallelLinear> fc2;
    std::shared_ptr<DropoutLayer> dropout;

public:
    DistributedGPTMLP(uint32_t embedding_size, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

class DistributedGPTBlock : public autograd::ModuleBase {
    std::shared_ptr<DistributedGPTMLP> mlp;
    std::shared_ptr<LayerNormLayer> ln1;
    std::shared_ptr<LayerNormLayer> ln2;
    std::shared_ptr<DistributedMultiHeadAttention> attention;

public:
    explicit DistributedGPTBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm = false);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules::distributed
