// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/llama_block.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::distributed::llama {

using RunnerType = common::transformer::RunnerType;

struct LlamaConfig {
    uint32_t num_heads = 6U;
    uint32_t num_groups = 3U;
    uint32_t embedding_dim = 384U;  // embedding dimension, must be divisible by num_heads
    float dropout_prob = 0.0F;
    uint32_t num_blocks = 6U;
    uint32_t vocab_size = 256U;
    uint32_t max_sequence_length = 256U;
    RunnerType runner_type = RunnerType::Default;
};

class DistributedLlama : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::autograd::ModuleBase> tok_emb;
    std::vector<std::shared_ptr<ModuleBase>> blocks;
    std::shared_ptr<ModuleBase> ln_fc;
    std::shared_ptr<ttml::autograd::ModuleBase> fc;
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit DistributedLlama(const LlamaConfig& config);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);
};

[[nodiscard]] LlamaConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const LlamaConfig& llama_config);
[[nodiscard]] std::shared_ptr<DistributedLlama> create(const LlamaConfig& config);
[[nodiscard]] std::shared_ptr<DistributedLlama> create(const YAML::Node& config);

}  // namespace ttml::models::distributed::llama
