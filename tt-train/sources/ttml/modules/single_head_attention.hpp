// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

class SingleHeadAttention : public ttml::modules::ModuleBase {
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ModuleBasePtr q_linear;
    ModuleBasePtr k_linear;
    ModuleBasePtr v_linear;
    ModuleBasePtr out_linear;
    ModuleBasePtr dropout;

public:
    explicit SingleHeadAttention(uint32_t embedding_dim, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
