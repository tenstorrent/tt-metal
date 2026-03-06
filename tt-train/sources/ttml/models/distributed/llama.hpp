// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "models/llama.hpp"
#include "modules/distributed/llama_block.hpp"
#include "modules/module_base.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::distributed::llama {

using RunnerType = common::transformer::RunnerType;
using models::llama::LlamaConfig;

struct PipelineParallelConfig {
    uint32_t num_blocks = 6U;
    std::unordered_map<uint32_t, uint32_t> blocks_per_rank;

    // void verify() const;
};

class DistributedLlama : public BaseTransformer {
private:
    bool is_first_rank() const;
    bool is_last_rank() const;
    uint32_t get_blocks_to_skip() const;
    uint32_t get_blocks_to_load(uint32_t) const;

    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::modules::ModuleBase> tok_emb;
    std::vector<std::shared_ptr<ttml::modules::distributed::DistributedLlamaBlock>> blocks;
    std::shared_ptr<ModuleBase> ln_fc;
    std::shared_ptr<ttml::modules::ModuleBase> fc;
    ops::RotaryEmbeddingParams m_rope_params;

    uint32_t embedding_dim{};
    std::optional<PipelineParallelConfig> pipeline_parallel_config;

public:
    explicit DistributedLlama(
        const LlamaConfig& config, const std::optional<PipelineParallelConfig>& pipeline_parallel_config);
    virtual ~DistributedLlama() = default;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const std::optional<ttml::autograd::TensorPtr>& mask) override;
};

[[nodiscard]] std::shared_ptr<DistributedLlama> create(
    const LlamaConfig& config, const std::optional<PipelineParallelConfig>& pipeline_parallel_config = std::nullopt);
[[nodiscard]] std::shared_ptr<DistributedLlama> create(
    const YAML::Node& config, const std::optional<PipelineParallelConfig>& pipeline_parallel_config = std::nullopt);
[[nodiscard]] PipelineParallelConfig read_config(const YAML::Node& config);

}  // namespace ttml::models::distributed::llama
