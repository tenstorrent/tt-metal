// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "autograd/module_base.hpp"
#include "models/common/transformer_common.hpp"

namespace ttml::models::gpt2 {

using RunnerType = common::transformer::RunnerType;
using WeightTyingType = common::transformer::WeightTyingType;

enum class PositionalEmbeddingType {
    Trainable,
    Fixed,
};

struct TransformerConfig {
    uint32_t num_heads = 6;
    uint32_t embedding_dim = 384;
    float dropout_prob = 0.2F;
    uint32_t num_blocks = 6;
    uint32_t vocab_size = 256;
    uint32_t max_sequence_length = 256;
    RunnerType runner_type = RunnerType::Default;
    WeightTyingType weight_tying = WeightTyingType::Disabled;
    PositionalEmbeddingType positional_embedding_type = PositionalEmbeddingType::Trainable;

    struct Experimental {
        bool use_composite_layernorm = false;
    };
    Experimental experimental;
};

class Transformer : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::autograd::ModuleBase> tok_emb;
    std::shared_ptr<ttml::autograd::ModuleBase> pos_emb;
    std::vector<std::shared_ptr<ttml::autograd::ModuleBase>> blocks;
    std::shared_ptr<ttml::autograd::ModuleBase> ln_fc;
    std::shared_ptr<ttml::autograd::ModuleBase> fc;

public:
    explicit Transformer(const TransformerConfig& config);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);
};

[[nodiscard]] TransformerConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const TransformerConfig& mlp_config);
[[nodiscard]] std::shared_ptr<Transformer> create(const TransformerConfig& config);
[[nodiscard]] std::shared_ptr<Transformer> create(const YAML::Node& config);

}  // namespace ttml::models::gpt2
