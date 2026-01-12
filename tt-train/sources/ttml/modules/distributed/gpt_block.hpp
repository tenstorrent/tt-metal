// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

class DistributedGPTMLP : public modules::ModuleBase {
public:
    DistributedGPTMLP(uint32_t embedding_size, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;

private:
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr m_fc1;
    ModuleBasePtr m_fc2;
    ModuleBasePtr m_dropout;
};

class DistributedGPTBlock : public modules::ModuleBase {
public:
    explicit DistributedGPTBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm = false);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;

private:
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr m_mlp;
    ModuleBasePtr m_ln1;
    ModuleBasePtr m_ln2;
    ModuleBasePtr m_attention;
};

}  // namespace ttml::modules::distributed
