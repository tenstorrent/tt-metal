// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "autograd/module_base.hpp"
#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"

struct TransformerConfig {
    uint32_t num_heads = 6;
    uint32_t embedding_dim = 384;
    float dropout_prob = 0.2F;
    uint32_t num_blocks = 6;
    uint32_t vocab_size = 256;
    uint32_t max_sequence_length = 256;
};

class Transformer : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::Embedding> tok_emb;
    std::shared_ptr<ttml::modules::Embedding> pos_emb;
    std::vector<std::shared_ptr<ttml::modules::GPTBlock>> blocks;
    std::shared_ptr<ttml::modules::LayerNormLayer> ln_fc;
    std::shared_ptr<ttml::modules::LinearLayer> fc;

public:
    explicit Transformer(const TransformerConfig& config);

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x,
        const ttml::autograd::TensorPtr& positions,
        const ttml::autograd::TensorPtr& mask);
};

class BigramFCModel : public ttml::autograd::ModuleBase {
public:
    std::shared_ptr<ttml::modules::LinearLayer> fc1;
    std::shared_ptr<ttml::modules::Embedding> emb;

    BigramFCModel(uint32_t vocab_size, uint32_t num_tokens, uint32_t hidden_dim);

    ttml::autograd::TensorPtr operator()(
        ttml::autograd::TensorPtr x,
        [[maybe_unused]] const ttml::autograd::TensorPtr& positions,
        [[maybe_unused]] const ttml::autograd::TensorPtr& masks) const;
};
