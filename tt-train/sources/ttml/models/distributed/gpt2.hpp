// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "modules/distributed/gpt_block.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/embedding_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/positional_embeddings.hpp"

namespace ttml::models::distributed::gpt2 {

enum class PositionalEmbeddingType {
    Trainable,
    Fixed,
};

enum class RunnerType {
    MemoryEfficient,
    Default,
};

enum class WeightTyingType {
    Disabled,
    Enabled,
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

class DistributedTransformer : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::modules::Embedding> tok_emb;
    std::shared_ptr<ttml::modules::PositionalEmbeddingBase> pos_emb;
    std::vector<std::shared_ptr<ttml::modules::distributed::DistributedGPTBlock>> blocks;
    std::shared_ptr<ttml::modules::LayerNormLayer> ln_fc;
    std::shared_ptr<ttml::modules::distributed::ColumnParallelLinear> fc;

public:
    explicit DistributedTransformer(const TransformerConfig& config);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);
};

[[nodiscard]] TransformerConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const TransformerConfig& mlp_config);
[[nodiscard]] std::shared_ptr<DistributedTransformer> create(const TransformerConfig& config);
[[nodiscard]] std::shared_ptr<DistributedTransformer> create(const YAML::Node& config);

}  // namespace ttml::models::distributed::gpt2
