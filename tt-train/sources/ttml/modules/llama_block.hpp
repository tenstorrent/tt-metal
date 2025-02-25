// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "modules/rms_norm_module.hpp"
#include "modules/single_head_attention.hpp"

namespace ttml::modules {

class LlamaMLP : public autograd::ModuleBase {
    std::shared_ptr<LinearLayer> w1;
    std::shared_ptr<LinearLayer> w3;
    std::shared_ptr<LinearLayer> w2;
    std::shared_ptr<DropoutLayer> dropout;

public:
    LlamaMLP(uint32_t embedding_size, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input);
};

class LlamaBlock : public autograd::ModuleBase {
    std::shared_ptr<LlamaMLP> mlp;
    std::shared_ptr<RMSNormLayer> attention_norm;
    std::shared_ptr<RMSNormLayer> ffn_norm;
    std::shared_ptr<MultiHeadAttention> attention;

public:
    explicit LlamaBlock(
        uint32_t embedding_size, uint32_t num_heads, float dropout_prob = 0.0F, bool use_composite_layernorm = false);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules
