// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt2.hpp"

#include "autograd/graph_utils.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/distributed_mapping.hpp"
#include "core/scoped.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/distributed/gpt_block.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/gpt_block.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::distributed::gpt2 {

namespace {

void weights_initialization(DistributedTransformer& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            auto tensor_shape = tensor.logical_shape();
            auto* device = &autograd::ctx().get_device();
            auto num_devices = static_cast<uint32_t>(device->num_devices());
            tensor_shape[0] *= num_devices;
            auto weight_xtensor = init::normal_init(tensor_shape, {0.F, 0.02F});
            const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
            tensor_ptr->set_value(ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
                weight_xtensor, device, ttnn::Layout::TILE, mapper.get()));
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.logical_shape(), 0.F);
        }
    }
}

}  // namespace

DistributedTransformer::DistributedTransformer(const TransformerConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    auto position_embedding_type = config.positional_embedding_type;
    auto use_composite_layernorm = config.experimental.use_composite_layernorm;
    runner_type = config.runner_type;

    fmt::print("DistributedTransformer configuration:\n");
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
        blocks.push_back(std::make_shared<ttml::modules::distributed::DistributedGPTBlock>(
            embedding_dim, num_heads, dropout_prob, use_composite_layernorm));
    }
    ln_fc = std::make_shared<ttml::modules::LayerNormLayer>(embedding_dim, use_composite_layernorm);
    fc = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
        embedding_dim, vocab_size, /* bias */ false, /* gather_output */ true);

    create_name("transformer");
    register_module(tok_emb, "tok_emb");
    register_module(pos_emb, "pos_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    if (config.weight_tying == WeightTyingType::Enabled) {
        throw std::logic_error("Weight tying is not supported yet for DistributedTransformer!");
    }

    weights_initialization(*this);
}

ttml::autograd::TensorPtr DistributedTransformer::operator()(
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

std::shared_ptr<DistributedTransformer> create(const TransformerConfig& config) {
    return std::make_shared<DistributedTransformer>(config);
}

std::shared_ptr<DistributedTransformer> create(const YAML::Node& config) {
    TransformerConfig transformer_config = models::gpt2::read_config(config);
    return std::make_shared<DistributedTransformer>(transformer_config);
}

}  // namespace ttml::models::distributed::gpt2
