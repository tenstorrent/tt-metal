// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer.hpp"

#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/llama_block.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::transformer {
RunnerType read_runner_type(const YAML::Node& config) {
    auto runner_type_str = config["runner_type"].as<std::string>("default");
    if (runner_type_str == "default") {
        return RunnerType::Default;
    } else if (runner_type_str == "memory_efficient") {
        return RunnerType::MemoryEfficient;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown runner type: {}. Supported runner types [default, memory_efficient]", runner_type_str));
    }
}

PositionalEmbeddingType read_positional_embedding_type(const YAML::Node& config) {
    auto positional_embedding_str = config["positional_embedding_type"].as<std::string>("trainable");
    if (positional_embedding_str == "trainable") {
        return PositionalEmbeddingType::Trainable;
    } else if (positional_embedding_str == "fixed") {
        return PositionalEmbeddingType::Fixed;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown positional embedding type: {}. Supported positional embedding types [trainable, fixed]",
            positional_embedding_str));
    }
}

WeightTyingType read_weight_tying_type(const YAML::Node& config) {
    auto weight_tying_str = config["weight_tying"].as<std::string>("disabled");
    if (weight_tying_str == "disabled") {
        return WeightTyingType::Disabled;
    } else if (weight_tying_str == "enabled") {
        return WeightTyingType::Enabled;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown weight tying type: {}. Supported weight tying types [disabled, enabled]", weight_tying_str));
    }
}

TransformerConfig read_config(const YAML::Node& config) {
    TransformerConfig transformer_config;
    transformer_config.num_heads = config["num_heads"].as<uint32_t>();
    transformer_config.embedding_dim = config["embedding_dim"].as<uint32_t>();
    transformer_config.dropout_prob = config["dropout_prob"].as<float>();
    transformer_config.num_blocks = config["num_blocks"].as<uint32_t>();
    transformer_config.vocab_size = config["vocab_size"].as<uint32_t>();
    transformer_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>();
    transformer_config.positional_embedding_type = read_positional_embedding_type(config);
    transformer_config.runner_type = read_runner_type(config);
    transformer_config.weight_tying = read_weight_tying_type(config);

    if (auto experimental_config = config["experimental"]) {
        transformer_config.experimental.use_composite_layernorm =
            experimental_config["use_composite_layernorm"].as<bool>();
    }
    return transformer_config;
}

YAML::Node write_config(const TransformerConfig& mlp_config) {
    YAML::Node config;
    config["num_heads"] = mlp_config.num_heads;
    config["embedding_dim"] = mlp_config.embedding_dim;
    config["dropout_prob"] = mlp_config.dropout_prob;
    config["num_blocks"] = mlp_config.num_blocks;
    config["vocab_size"] = mlp_config.vocab_size;
    config["max_sequence_length"] = mlp_config.max_sequence_length;
    return config;
}

}  // namespace ttml::models::transformer
