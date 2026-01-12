// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "modules/rms_norm_module.hpp"
#include "modules/single_head_attention.hpp"

namespace ttml::modules {

class GPTMLP : public modules::ModuleBase {
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr fc1;
    ModuleBasePtr fc2;
    ModuleBasePtr dropout;

public:
    GPTMLP(uint32_t embedding_size, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

class GPTBlock : public modules::ModuleBase {
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr mlp;
    ModuleBasePtr ln1;
    ModuleBasePtr ln2;
    ModuleBasePtr attention;

public:
    explicit GPTBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm = false);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
