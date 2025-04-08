// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama.hpp"

#include "autograd/tensor.hpp"
#include "core/distributed_mapping.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/distributed/llama_block.hpp"
#include "modules/embedding_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::distributed::llama {

namespace {

void initialize_weights(DistributedLlama& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            auto tensor_shape = tensor.get_logical_shape();
            auto* device = &autograd::ctx().get_device();
            auto num_devices = static_cast<uint32_t>(device->num_devices());
            tensor_shape[0] *= num_devices;
            core::XTensorToMeshVariant<float> shard_composer = core::ShardXTensorToMesh<float>(device->shape(), 0);
            auto weight_xtensor = init::normal_init(tensor_shape, {0.F, 0.02F});
            tensor_ptr->set_value(
                core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight_xtensor, device, shard_composer));
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.get_logical_shape(), 0.F);
        }
    }
}

}  // namespace

DistributedLlama::DistributedLlama(const LlamaConfig& config) {
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

    m_rope_params = ops::build_rope_params(max_sequence_length, embedding_dim / num_heads);
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<modules::distributed::DistributedLlamaBlock>(
            embedding_dim, num_heads, num_groups, m_rope_params, dropout_prob));
    }
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
    fc = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
        embedding_dim, vocab_size, /* has_bias */ false, /* gather_output */ true);

    create_name("llama");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("llama_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    initialize_weights(*this);
}

autograd::TensorPtr DistributedLlama::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
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
    auto log_softmax = ttml::ops::log_softmax_moreh(logits, 3);
    return log_softmax;
}

std::shared_ptr<DistributedLlama> create(const LlamaConfig& config) {
    return std::make_shared<DistributedLlama>(config);
}
std::shared_ptr<DistributedLlama> create(const YAML::Node& config) {
    LlamaConfig llama_config = models::llama::read_config(config);
    return std::make_shared<DistributedLlama>(llama_config);
}

}  // namespace ttml::models::distributed::llama
