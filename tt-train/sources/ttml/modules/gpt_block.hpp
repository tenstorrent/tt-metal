// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

class GPTMLP : public autograd::ModuleBase {
    std::shared_ptr<LinearLayer> fc1;
    std::shared_ptr<LinearLayer> fc2;
    std::shared_ptr<DropoutLayer> dropout;

public:
    GPTMLP(uint32_t embedding_size, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

class GPTBlock : public autograd::ModuleBase {
    std::shared_ptr<GPTMLP> mlp;
    std::shared_ptr<LayerNormLayer> ln1;
    std::shared_ptr<LayerNormLayer> ln2;
    std::shared_ptr<MultiHeadAttention> attention;

public:
    explicit GPTBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob, bool use_composite_layernorm = false);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules
