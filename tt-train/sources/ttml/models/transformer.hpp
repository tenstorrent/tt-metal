// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <concepts>
#include <type_traits>

#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::transformer {

// Runtime configuration enums (still needed for YAML parsing)
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

// Define concepts for component types

// Concept for normalization layers
template <typename T>
concept NormLayer = requires(T t, const ttml::autograd::TensorPtr& x) {
    { t(x) } -> std::convertible_to<ttml::autograd::TensorPtr>;
    requires std::derived_from<T, ttml::autograd::ModuleBase>;
};

// Concept for transformer blocks
template <typename T>
concept TransformerBlock = requires(T t, const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    { t(x, mask) } -> std::convertible_to<ttml::autograd::TensorPtr>;
    requires std::derived_from<T, ttml::autograd::ModuleBase>;
};

// Base configuration struct
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

// Templatized transformer implementation
template <TransformerBlock BlockT, NormLayer NormLayerT>
class Transformer : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type;
    std::shared_ptr<ttml::modules::Embedding> tok_emb;
    std::shared_ptr<ttml::modules::PositionalEmbeddingBase> pos_emb;
    std::vector<std::shared_ptr<BlockT>> blocks;
    std::shared_ptr<NormLayerT> ln_fc;
    std::shared_ptr<ttml::modules::LinearLayer> fc;

public:
    explicit Transformer(const TransformerConfig& config);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);
};

[[nodiscard]] TransformerConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const TransformerConfig& config);

template <TransformerBlock BlockT, NormLayer NormLayerT>
[[nodiscard]] std::shared_ptr<Transformer<BlockT, NormLayerT>> create(const TransformerConfig& config);

template <TransformerBlock BlockT, NormLayer NormLayerT>
[[nodiscard]] std::shared_ptr<Transformer<BlockT, NormLayerT>> create(const YAML::Node& config);

}  // namespace ttml::models::transformer

// impls
namespace ttml::models::transformer {
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

template <TransformerBlock BlockT, NormLayer NormLayerT>
void weights_initialization(Transformer<BlockT, NormLayerT>& model) {
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

template <TransformerBlock BlockT, NormLayer NormLayerT>
Transformer<BlockT, NormLayerT>::Transformer(const TransformerConfig& config) {
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
    tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);

    auto create_positional_embedding = [position_embedding_type,
                                        max_sequence_length,
                                        embedding_dim,
                                        dropout_prob]() -> std::shared_ptr<modules::PositionalEmbeddingBase> {
        if (position_embedding_type == PositionalEmbeddingType::Trainable) {
            return std::make_shared<ttml::modules::TrainablePositionalEmbedding>(
                embedding_dim, dropout_prob, max_sequence_length);
        } else {
            return std::make_shared<ttml::modules::PositionalEmbedding>(
                embedding_dim, dropout_prob, max_sequence_length);
        }
    };
    pos_emb = create_positional_embedding();
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<BlockT>(embedding_dim, num_heads, dropout_prob, use_composite_layernorm));
    }
    ln_fc = std::make_shared<NormLayerT>(embedding_dim);
    fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size, /* bias */ false);

    create_name("transformer");
    register_module(tok_emb, "tok_emb");
    register_module(pos_emb, "pos_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("transformer_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    if (config.weight_tying == WeightTyingType::Enabled) {
        // tie weights between embedding and fc
        tok_emb->set_weight(fc->get_weight());
    }

    weights_initialization(*this);
}

template <TransformerBlock BlockT, NormLayer NormLayerT>
ttml::autograd::TensorPtr Transformer<BlockT, NormLayerT>::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = (*pos_emb)(tok_emb_out);
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

template <TransformerBlock BlockT, NormLayer NormLayerT>
std::shared_ptr<Transformer<BlockT, NormLayerT>> create(const TransformerConfig& config) {
    return std::make_shared<Transformer<BlockT, NormLayerT>>(config);
}

template <TransformerBlock BlockT, NormLayer NormLayerT>
std::shared_ptr<Transformer<BlockT, NormLayerT>> create(const YAML::Node& config) {
    TransformerConfig transformer_config = read_config(config);
    return std::make_shared<Transformer<BlockT, NormLayerT>>(transformer_config);
}

}  // namespace ttml::models::transformer
