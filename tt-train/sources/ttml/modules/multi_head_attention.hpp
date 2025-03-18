// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

class MultiHeadAttention : public ttml::autograd::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    std::shared_ptr<LinearLayer> m_qkv_linear;
    std::shared_ptr<LinearLayer> m_out_linear;
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    explicit MultiHeadAttention(uint32_t embedding_dim, uint32_t num_heads, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
