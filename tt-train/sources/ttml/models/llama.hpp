// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "common/transformer_common.hpp"
#include "modules/llama_block.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::llama {

using RunnerType = common::transformer::RunnerType;
using WeightTyingType = common::transformer::WeightTyingType;

struct LlamaConfig {
    uint32_t num_heads = 6U;
    uint32_t num_groups = 3U;
    uint32_t embedding_dim = 384U;  // embedding dimension, must be divisible by num_heads
    std::optional<uint32_t> intermediate_dim = std::nullopt;
    float dropout_prob = 0.0F;
    float theta = 10000.0F;
    uint32_t num_blocks = 6U;
    uint32_t vocab_size = 256U;
    uint32_t max_sequence_length = 256U;
    RunnerType runner_type = RunnerType::Default;
    WeightTyingType weight_tying = WeightTyingType::Disabled;

    // RoPE NTK-aware scaling parameters
    float scaling_factor = 0.0F;  // 0.0 means no scaling
    float high_freq_factor = 4.0F;
    float low_freq_factor = 1.0F;
    uint32_t original_context_length = 0U;
};

class Llama : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::autograd::ModuleBase> tok_emb;
    std::vector<std::shared_ptr<ModuleBase>> blocks;
    std::shared_ptr<ModuleBase> ln_fc;
    std::shared_ptr<ttml::autograd::ModuleBase> fc;
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit Llama(const LlamaConfig& config);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask);
};

[[nodiscard]] LlamaConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const LlamaConfig& llama_config);
[[nodiscard]] std::shared_ptr<Llama> create(const LlamaConfig& config);
[[nodiscard]] std::shared_ptr<Llama> create(const YAML::Node& config);

}  // namespace ttml::models::llama
