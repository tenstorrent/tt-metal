// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/distributed/qwen_block.hpp"
#include "modules/embedding_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::distributed::qwen {

namespace {

void initialize_weights(DistributedQwen& model) {
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

DistributedQwen::DistributedQwen(const QwenConfig& config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    uint32_t num_groups = config.num_groups;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    runner_type = config.runner_type;
    float theta = config.theta;
    auto intermediate_dim = config.intermediate_dim;

    fmt::print("Qwen configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Theta: {}\n", theta);
    if (intermediate_dim) {
        fmt::print("    Intermediate dim: {}\n", *intermediate_dim);
    } else {
        fmt::print("    Intermediate dim: {} (4 * embedding_dim)\n", 4 * embedding_dim);
    }

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
        blocks.push_back(std::make_shared<modules::distributed::DistributedQwenBlock>(
            embedding_dim, num_heads, num_groups, m_rope_params, dropout_prob, intermediate_dim));
    }
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
    fc = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
        embedding_dim, vocab_size, /* has_bias */ true, /* gather_output */ true);

    create_name("qwen");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("qwen_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    initialize_weights(*this);
}

autograd::TensorPtr DistributedQwen::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;  // qwen does positional embedding in the attention blocks
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

std::shared_ptr<DistributedQwen> create(const QwenConfig& config) {
    return std::make_shared<DistributedQwen>(config);
}
std::shared_ptr<DistributedQwen> create(const YAML::Node& config) {
    QwenConfig qwen_config = models::qwen::read_config(config);
    return std::make_shared<DistributedQwen>(qwen_config);
}

}  // namespace ttml::models::distributed::qwen
