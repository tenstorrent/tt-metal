// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt2.hpp"

#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::gpt2 {

Transformer::Transformer(const TransformerConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    auto position_embedding_type = config.positional_embedding_type;
    auto use_composite_layernorm = config.experimental.use_composite_layernorm;
    runner_type = config.runner_type;

    fmt::print("Transformer configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print(
        "    Positional embedding type: {}\n",
        position_embedding_type == PositionalEmbeddingType::Trainable ? "Trainable" : "Fixed");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Composite layernorm: {}\n", use_composite_layernorm);
    fmt::print("    Weight tying: {}\n", config.weight_tying == WeightTyingType::Enabled ? "Enabled" : "Disabled");

    uint32_t vocab_size_divisible_by_32 = (vocab_size + 31) / 32 * 32;
    if (max_sequence_length % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Max sequence length should be divisible by 32 due to current limitations in tensor. Provided "
            "max_sequence_length={}",
            max_sequence_length));
    }
    if (embedding_dim % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Embedding size should be divisible by 32 due to current limitations in tensor. Provided "
            "embedding_dim={}",
            embedding_dim));
    }
    auto last_fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size, /* bias */ false);
    if (config.weight_tying == WeightTyingType::Enabled) {
        tok_emb = std::make_shared<ttml::modules::Embedding>(last_fc->get_weight());
    } else {
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    }

    auto create_positional_embedding = [position_embedding_type,
                                        max_sequence_length,
                                        embedding_dim,
                                        dropout_prob]() -> std::shared_ptr<autograd::ModuleBase> {
        if (position_embedding_type == PositionalEmbeddingType::Trainable) {
            return std::make_shared<ttml::modules::TrainablePositionalEmbedding>(
                ttml::modules::PositionalEmbeddingConfig{
                    .embedding_dim = embedding_dim,
                    .sequence_length = max_sequence_length,
                    .dropout_prob = dropout_prob,
                    .use_dropout_seed_per_device = false});
        } else {
            return std::make_shared<ttml::modules::PositionalEmbedding>(ttml::modules::PositionalEmbeddingConfig{
                .embedding_dim = embedding_dim,
                .sequence_length = max_sequence_length,
                .dropout_prob = dropout_prob,
                .use_dropout_seed_per_device = false});
        }
    };
    pos_emb = create_positional_embedding();
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(
            std::make_shared<ttml::modules::GPTBlock>(embedding_dim, num_heads, dropout_prob, use_composite_layernorm));
    }
    ln_fc = std::make_shared<ttml::modules::LayerNormLayer>(embedding_dim, use_composite_layernorm);
    fc = last_fc;

    create_name("transformer");
    register_module(tok_emb, "tok_emb");
    register_module(pos_emb, "pos_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    common::transformer::initialize_weights_gpt2(*this);
}

ttml::autograd::TensorPtr Transformer::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = (*pos_emb)(tok_emb_out);
    for (auto& block : blocks) {
        if (runner_type == RunnerType::MemoryEfficient) {
            out = common::transformer::memory_efficient_runner(*block, out, mask);
        } else if (runner_type == RunnerType::Default) {
            out = (*block)(out, mask);
        } else {
            throw std::runtime_error("Unknown runner type. Supported runner types ['default', 'memory_efficient']");
        }
    }
    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    return logits;
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

TransformerConfig read_config(const YAML::Node& config) {
    TransformerConfig transformer_config;
    transformer_config.num_heads = config["num_heads"].as<uint32_t>();
    transformer_config.embedding_dim = config["embedding_dim"].as<uint32_t>();
    transformer_config.dropout_prob = config["dropout_prob"].as<float>();
    transformer_config.num_blocks = config["num_blocks"].as<uint32_t>();
    transformer_config.vocab_size = config["vocab_size"].as<uint32_t>();
    transformer_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>();
    transformer_config.positional_embedding_type = read_positional_embedding_type(config);
    transformer_config.runner_type = common::transformer::read_runner_type(config);
    transformer_config.weight_tying = common::transformer::read_weight_tying_type(config);

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

std::shared_ptr<Transformer> create(const TransformerConfig& config) {
    return std::make_shared<Transformer>(config);
}
std::shared_ptr<Transformer> create(const YAML::Node& config) {
    TransformerConfig transformer_config = read_config(config);
    return std::make_shared<Transformer>(transformer_config);
}

}  // namespace ttml::models::gpt2
