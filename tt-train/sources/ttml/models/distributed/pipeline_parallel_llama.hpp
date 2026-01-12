// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "llama.hpp"
#include "models/common/transformer_common.hpp"
#include "models/llama.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/distributed/llama_block.hpp"
#include "modules/llama_block.hpp"
#include "ops/rope_op.hpp"

namespace ttml::models::distributed::pipeline_parallel_llama {

using RunnerType = common::transformer::RunnerType;
using models::llama::LlamaConfig;

struct PipelineParallelConfig {
    uint32_t num_blocks = 6U;
    std::unordered_map<uint32_t, uint32_t> blocks_per_rank;

    void verify() const;
};

class PipelineParallelLlama : public BaseTransformer {
public:
    PipelineParallelLlama(
        const LlamaConfig& config,
        const PipelineParallelConfig& pipeline_parallel_config,
        bool is_tensor_parallel = false);
    virtual ~PipelineParallelLlama() = default;
    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);

private:
    bool is_first_rank() const;
    bool is_last_rank() const;
    uint32_t get_blocks_to_skip() const;
    uint32_t get_blocks_to_load() const;

    RunnerType runner_type = RunnerType::Default;
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ttml::modules::ModuleBasePtr tok_emb;
    std::vector<ttml::modules::ModuleBasePtr> blocks;
    ttml::modules::ModuleBasePtr ln_fc;
    ttml::modules::ModuleBasePtr fc;
    ops::RotaryEmbeddingParams rope_params;

    uint32_t embedding_dim{};
    PipelineParallelConfig pipeline_parallel_config;
};

[[nodiscard]] PipelineParallelConfig read_config(const YAML::Node& config);
[[nodiscard]] std::shared_ptr<PipelineParallelLlama> create(
    const LlamaConfig& config, const PipelineParallelConfig& pipeline_parallel_config, bool is_tensor_parallel = false);

}  // namespace ttml::models::distributed::pipeline_parallel_llama
