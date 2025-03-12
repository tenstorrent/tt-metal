// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama.hpp"

#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/embedding_module.hpp"
#include "modules/llama_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::llama {

namespace {

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl, const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return forward_impl(input, mask);
    }

    // make a copy of a generator before running forward pass
    auto generator = autograd::ctx().get_generator();

    // running forward pass
    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = forward_impl(input, mask);
    }

    // define grad function and copy generator (in the state before forward pass)
    autograd::GradFunction grad = [input, mask, out, &forward_impl, generator]() {
        // detach input from existing graph
        auto input_detached = autograd::create_tensor(input->get_value());
        // run forward pass again
        autograd::TensorPtr output;
        {
            // set generator to the state before forward pass during construction
            // restore generator state after grad function is executed
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = forward_impl(input_detached, mask);
        }
        // use gradients from new output
        output->set_grad(out->get_grad());
        output->backward();
        // reuse gradients from detached input
        input->add_grad(input_detached->get_grad());
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

void weights_initialization(Llama& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            init::normal_init(tensor_ptr, tensor.get_logical_shape(), {0.F, 0.02F});
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.get_logical_shape(), 0.F);
        }
    }
}

}  // namespace

Llama::Llama(const LlamaConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    uint32_t num_groups = config.num_groups;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    runner_type = config.runner_type;

    fmt::print("Llama configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
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

    m_rope_params = ops::build_rope_params(max_sequence_length, embedding_dim / num_heads);
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::LlamaBlock>(
            embedding_dim, num_heads, num_groups, dropout_prob, &m_rope_params));
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

    weights_initialization(*this);
}

ttml::autograd::TensorPtr Llama::operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;  // llama does positional embedding in the attention blocks
    for (auto& block : blocks) {
        if (runner_type == RunnerType::MemoryEfficient) {
            out = memory_efficient_runner(*block, out, mask);
        } else if (runner_type == RunnerType::Default) {
            out = (*block)(out, mask);
        } else {
            throw std::runtime_error("Unknown runner type. Supported runner types ['default', 'memory_efficient']");
        }
    }
    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    auto log_softmax = ttml::ops::log_softmax_moreh(logits, 3);
    return log_softmax;
}

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

LlamaConfig read_config(const YAML::Node& config) {
    LlamaConfig llama_config;
    llama_config.num_heads = config["num_heads"].as<uint32_t>();
    // FIXME: add num_groups to config
    llama_config.embedding_dim = config["embedding_dim"].as<uint32_t>();
    llama_config.dropout_prob = config["dropout_prob"].as<float>();
    llama_config.num_blocks = config["num_blocks"].as<uint32_t>();
    llama_config.vocab_size = config["vocab_size"].as<uint32_t>();
    llama_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>();
    llama_config.runner_type = read_runner_type(config);
    llama_config.weight_tying = read_weight_tying_type(config);

    return llama_config;
}

YAML::Node write_config(const LlamaConfig& llama_config) {
    YAML::Node config;
    config["num_heads"] = llama_config.num_heads;
    config["embedding_dim"] = llama_config.embedding_dim;
    config["dropout_prob"] = llama_config.dropout_prob;
    config["num_blocks"] = llama_config.num_blocks;
    config["vocab_size"] = llama_config.vocab_size;
    config["max_sequence_length"] = llama_config.max_sequence_length;
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
