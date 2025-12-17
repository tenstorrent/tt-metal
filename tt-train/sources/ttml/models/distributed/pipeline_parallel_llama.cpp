// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_parallel_llama.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/distributed/pipeline_parallel_comm_ops.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::distributed::pipeline_parallel_llama {

namespace {

void initialize_weights_tensor_parallel(PipelineParallelLlama& model) {
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

void PipelineParallelConfig::verify() const {
    auto total_blocks = std::accumulate(
        blocks_per_rank.begin(), blocks_per_rank.end(), 0, [](int sum, const auto& pair) { return sum + pair.second; });
    if (num_blocks != total_blocks) {
        throw std::runtime_error("Number of blocks must match number of blocks per rank.");
    }
}

PipelineParallelConfig read_config(const YAML::Node& config) {
    PipelineParallelConfig pipeline_parallel_config;
    pipeline_parallel_config.num_blocks = config["num_blocks"].as<uint32_t>();
    pipeline_parallel_config.blocks_per_rank = config["blocks_per_rank"].as<std::unordered_map<uint32_t, uint32_t>>();
    pipeline_parallel_config.verify();
    return pipeline_parallel_config;
}

PipelineParallelLlama::PipelineParallelLlama(
    const LlamaConfig& config, const PipelineParallelConfig& pipeline_parallel_config, bool is_tensor_parallel) :
    pipeline_parallel_config(pipeline_parallel_config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    this->embedding_dim = config.embedding_dim;
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
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Theta: {}\n", theta);

    fmt::println("  Pipeline parallel configuration:");
    fmt::println("    Num blocks: {}", pipeline_parallel_config.num_blocks);
    fmt::println("    Blocks per rank:");
    for (const auto& [rank, blocks] : pipeline_parallel_config.blocks_per_rank) {
        fmt::println("      Rank {}: {}", rank, blocks);
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
    rope_params = ops::build_rope_params(
        /*sequence_length=*/max_sequence_length,
        /*head_dim=*/embedding_dim / num_heads,
        /*theta=*/theta,
        /*rope_scaling_params=*/rope_scaling_params);

    if (is_first_rank()) {
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    }

    auto blocks_to_skip = get_blocks_to_skip();
    auto blocks_to_load = get_blocks_to_load();

    blocks.reserve(blocks_to_load);
    for (uint32_t block_idx = blocks_to_skip; block_idx < blocks_to_skip + blocks_to_load; ++block_idx) {
        if (is_tensor_parallel) {
            blocks.push_back(std::make_shared<ttml::modules::distributed::DistributedLlamaBlock>(
                embedding_dim, num_heads, num_groups, rope_params, dropout_prob));
        } else {
            blocks.push_back(std::make_shared<ttml::modules::LlamaBlock>(
                embedding_dim, num_heads, num_groups, rope_params, dropout_prob));
        }
    }

    if (is_last_rank()) {
        ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
        if (is_tensor_parallel) {
            fc = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
                embedding_dim, vocab_size, /* has_bias */ false, /* gather_output */ true);
        } else {
            fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size, /* bias */ false);
        }
    }

    create_name("pipeline_parallel_llama");
    if (tok_emb) {
        register_module(tok_emb, "tok_emb");
    }
    for (uint32_t block_idx = 0; block_idx < blocks_to_load; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("llama_block_{}", block_idx + blocks_to_skip));
    }
    if (ln_fc) {
        assert(fc != nullptr);
        register_module(ln_fc, "ln_fc");
        register_module(fc, "fc");
    }

    if (is_tensor_parallel) {
        initialize_weights_tensor_parallel(*this);
    } else {
        common::transformer::initialize_weights_gpt2(*this);
    }
}

bool PipelineParallelLlama::is_first_rank() const {
    auto distributed_ctx = autograd::ctx().get_distributed_context();
    int rank = *distributed_ctx->rank();
    return rank == 0;
}

bool PipelineParallelLlama::is_last_rank() const {
    auto distributed_ctx = autograd::ctx().get_distributed_context();
    int rank = *distributed_ctx->rank();
    int size = *distributed_ctx->size();
    return rank + 1 == size;
}

uint32_t PipelineParallelLlama::get_blocks_to_skip() const {
    auto distributed_ctx = autograd::ctx().get_distributed_context();
    int our_rank = *distributed_ctx->rank();
    auto blocks_to_skip = 0U;
    for (const auto& [rank_key, blocks] : pipeline_parallel_config.blocks_per_rank) {
        if (static_cast<int>(rank_key) < our_rank) {
            blocks_to_skip += blocks;
        }
    }
    return blocks_to_skip;
}

uint32_t PipelineParallelLlama::get_blocks_to_load() const {
    auto distributed_ctx = autograd::ctx().get_distributed_context();
    int our_rank = *distributed_ctx->rank();
    return pipeline_parallel_config.blocks_per_rank.at(static_cast<uint32_t>(our_rank));
}

autograd::TensorPtr PipelineParallelLlama::operator()(const autograd::TensorPtr& x, const autograd::TensorPtr& mask) {
    auto out = x;
    if (is_first_rank()) {
        out = (*tok_emb)(out);
    } else {
        auto distributed_ctx = autograd::ctx().get_distributed_context();
        int rank = *distributed_ctx->rank();
        auto recv_rank = core::distributed::Rank(rank - 1);

        auto batch_size = out->get_value().logical_shape()[0];
        auto seq_len = out->get_value().logical_shape()[-1];

        auto recv_tensor = ttml::core::empty(
            ttnn::Shape{batch_size, 1U, seq_len, embedding_dim}, &autograd::ctx().get_device(), ttnn::MemoryConfig{});
        out = autograd::create_tensor(recv_tensor);
        out = ttml::ops::distributed::intermesh_recv(out, recv_rank);
    }

    for (auto& block : blocks) {
        if (runner_type == RunnerType::MemoryEfficient) {
            out = common::transformer::memory_efficient_runner(*block, out, mask);
        } else if (runner_type == RunnerType::Default) {
            out = (*block)(out, mask);
        }
    }

    if (is_last_rank()) {
        out = (*ln_fc)(out);
        out = (*fc)(out);
    } else {
        auto distributed_ctx = autograd::ctx().get_distributed_context();
        int rank = *distributed_ctx->rank();
        auto send_rank = core::distributed::Rank(rank + 1);
        out = ttml::ops::distributed::intermesh_send(out, send_rank);
    }

    return out;
}

std::shared_ptr<PipelineParallelLlama> create(
    const LlamaConfig& config, const PipelineParallelConfig& pipeline_parallel_config, bool is_tensor_parallel) {
    return std::make_shared<PipelineParallelLlama>(config, pipeline_parallel_config, is_tensor_parallel);
}

}  // namespace ttml::models::distributed::pipeline_parallel_llama
