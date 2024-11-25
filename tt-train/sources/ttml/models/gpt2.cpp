// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt2.hpp"

#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::gpt2 {

Transformer::Transformer(const TransformerConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;

    fmt::print("Transformer configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);

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
    tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    pos_emb = std::make_shared<ttml::modules::Embedding>(max_sequence_length, embedding_dim);
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::GPTBlock>(embedding_dim, num_heads, dropout_prob));
    }
    ln_fc = std::make_shared<ttml::modules::LayerNormLayer>(embedding_dim);
    fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size);

    create_name("transformer");
    register_module(tok_emb, "tok_emb");
    register_module(pos_emb, "pos_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");
}

ttml::autograd::TensorPtr Transformer::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& positions,
    const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto pos_emb_out = (*pos_emb)(positions);
    auto out = ttml::ops::add(tok_emb_out, pos_emb_out);
    for (auto& block : blocks) {
        out = (*block)(out, mask);
    }
    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    auto log_softmax = ttml::ops::log_softmax(logits, 3);
    return log_softmax;
}

TransformerConfig read_config(const YAML::Node& config) {
    TransformerConfig transformer_config;
    transformer_config.num_heads = config["num_heads"].as<uint32_t>();
    transformer_config.embedding_dim = config["embedding_dim"].as<uint32_t>();
    transformer_config.dropout_prob = config["dropout_prob"].as<float>();
    transformer_config.num_blocks = config["num_blocks"].as<uint32_t>();
    transformer_config.vocab_size = config["vocab_size"].as<uint32_t>();
    transformer_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>();
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
