// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama.hpp"

#include "autograd/tensor.hpp"
#include "modules/embedding_module.hpp"
#include "modules/llama_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::llama {

Llama::Llama(const LlamaConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    std::optional<uint32_t> intermediate_dim = config.intermediate_dim;
    uint32_t num_heads = config.num_heads;
    uint32_t num_groups = config.num_groups;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    runner_type = config.runner_type;
    float theta = config.theta;

    fmt::print("Llama configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Intermediate dim: {}\n", intermediate_dim ? fmt::format("{}", *intermediate_dim) : "None");
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Weight tying: {}\n", config.weight_tying == WeightTyingType::Enabled ? "Enabled" : "Disabled");
    fmt::print("    Theta: {}\n", theta);

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

    // Create RoPE scaling params if they are set
    ops::RopeScalingParams rope_scaling_params;
    if (config.scaling_factor != 0.0F && config.original_context_length != 0U) {
        rope_scaling_params.original_context_length = config.original_context_length;
        rope_scaling_params.scaling_factor = config.scaling_factor;
        rope_scaling_params.high_freq_factor = config.high_freq_factor;
        rope_scaling_params.low_freq_factor = config.low_freq_factor;

        fmt::print("    RoPE scaling enabled:\n");
        fmt::print("        Scaling factor: {}\n", config.scaling_factor);
        fmt::print("        Original context length: {}\n", config.original_context_length);
        fmt::print("        High freq factor: {}\n", config.high_freq_factor);
        fmt::print("        Low freq factor: {}\n", config.low_freq_factor);
    }

    m_rope_params = ops::build_rope_params(
        /*sequence_length=*/max_sequence_length,
        /*head_dim=*/embedding_dim / num_heads,
        /*theta=*/theta,
        /*rope_scaling_params=*/rope_scaling_params);
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::LlamaBlock>(
            embedding_dim, num_heads, num_groups, m_rope_params, dropout_prob, intermediate_dim));
    }
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
    fc = last_fc;

    create_name("llama");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("llama_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    common::transformer::initialize_weights_gpt2(*this);
}

ttml::autograd::TensorPtr Llama::operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;  // llama does positional embedding in the attention blocks
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

LlamaConfig read_config(const YAML::Node& config) {
    LlamaConfig llama_config;
    // Use defaults from nanollama3
    llama_config.num_heads = config["num_heads"].as<uint32_t>(6U);
    llama_config.num_groups = config["num_groups"].as<uint32_t>(3U);
    llama_config.embedding_dim = config["embedding_dim"].as<uint32_t>(384U);
    if (config["intermediate_dim"]) {
        uint32_t intermediate_dim = config["intermediate_dim"].as<uint32_t>();
        llama_config.intermediate_dim = std::make_optional(intermediate_dim);
    }
    llama_config.dropout_prob = config["dropout_prob"].as<float>(0.0F);
    llama_config.num_blocks = config["num_blocks"].as<uint32_t>(6U);
    llama_config.vocab_size = config["vocab_size"].as<uint32_t>(96U);
    llama_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>(256U);
    llama_config.theta = config["theta"].as<float>(500000.0F);
    llama_config.runner_type = common::transformer::read_runner_type(config);
    llama_config.weight_tying = common::transformer::read_weight_tying_type(config);

    // Read RoPE NTK-aware scaling parameters if they exist
    if (config["rope_scaling"]) {
        const auto& rope_scaling = config["rope_scaling"];
        if (rope_scaling["scaling_factor"]) {
            llama_config.scaling_factor = rope_scaling["scaling_factor"].as<float>();
        }
        if (rope_scaling["high_freq_factor"]) {
            llama_config.high_freq_factor = rope_scaling["high_freq_factor"].as<float>();
        }
        if (rope_scaling["low_freq_factor"]) {
            llama_config.low_freq_factor = rope_scaling["low_freq_factor"].as<float>();
        }
        if (rope_scaling["original_context_length"]) {
            llama_config.original_context_length = rope_scaling["original_context_length"].as<uint32_t>();
        }
    }

    return llama_config;
}

YAML::Node write_config(const LlamaConfig& llama_config) {
    YAML::Node config;
    config["num_heads"] = llama_config.num_heads;
    config["num_groups"] = llama_config.num_groups;
    config["embedding_dim"] = llama_config.embedding_dim;
    if (llama_config.intermediate_dim) {
        config["intermediate_dim"] = *llama_config.intermediate_dim;
    }
    config["dropout_prob"] = llama_config.dropout_prob;
    config["num_blocks"] = llama_config.num_blocks;
    config["vocab_size"] = llama_config.vocab_size;
    config["max_sequence_length"] = llama_config.max_sequence_length;
    config["theta"] = llama_config.theta;

    // Add RoPE scaling parameters if they are set
    if (llama_config.scaling_factor != 0.0F && llama_config.original_context_length != 0U) {
        YAML::Node rope_scaling;
        rope_scaling["scaling_factor"] = llama_config.scaling_factor;
        rope_scaling["high_freq_factor"] = llama_config.high_freq_factor;
        rope_scaling["low_freq_factor"] = llama_config.low_freq_factor;
        rope_scaling["original_context_length"] = llama_config.original_context_length;
        config["rope_scaling"] = rope_scaling;
    }

    return config;
}

std::shared_ptr<Llama> create(const LlamaConfig& config) {
    return std::make_shared<Llama>(config);
}
std::shared_ptr<Llama> create(const YAML::Node& config) {
    LlamaConfig llama_config = read_config(config);
    return std::make_shared<Llama>(llama_config);
}

}  // namespace ttml::models::llama
