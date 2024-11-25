// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"

namespace ttml::models::gpt2 {

struct TransformerConfig {
    uint32_t num_heads = 6;
    uint32_t embedding_dim = 384;
    float dropout_prob = 0.2F;
    uint32_t num_blocks = 6;
    uint32_t vocab_size = 256;
    uint32_t max_sequence_length = 256;
};

class Transformer : public ttml::autograd::ModuleBase {
private:
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

[[nodiscard]] TransformerConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const TransformerConfig& mlp_config);
[[nodiscard]] std::shared_ptr<Transformer> create(const TransformerConfig& config);
[[nodiscard]] std::shared_ptr<Transformer> create(const YAML::Node& config);

}  // namespace ttml::models::gpt2
