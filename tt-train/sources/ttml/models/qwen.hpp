// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "base_transformer.hpp"
#include "common/transformer_common.hpp"
#include "modules/qwen_block.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::qwen {

using RunnerType = common::transformer::RunnerType;
using WeightTyingType = common::transformer::WeightTyingType;

struct QwenConfig {
    uint32_t num_heads = 32U;
    uint32_t num_groups = 32U;       // Qwen2 typically uses full attention (num_groups = num_heads)
    uint32_t embedding_dim = 4096U;  // embedding dimension, must be divisible by num_heads
    std::optional<uint32_t> intermediate_dim = std::nullopt;  // MLP intermediate dimension
    float dropout_prob = 0.0F;
    float theta = 10000.0F;
    uint32_t num_blocks = 32U;
    uint32_t vocab_size = 151665U;  // Qwen2 vocab size
    uint32_t max_sequence_length = 2048U;
    RunnerType runner_type = RunnerType::Default;
    WeightTyingType weight_tying = WeightTyingType::Disabled;

    // RoPE NTK-aware scaling parameters
    float scaling_factor = 0.0F;  // 0.0 means no scaling
    float high_freq_factor = 4.0F;
    float low_freq_factor = 1.0F;
    uint32_t original_context_length = 0U;
};

class Qwen : public BaseTransformer {
private:
    RunnerType runner_type = RunnerType::Default;
    QwenConfig m_config;
    std::shared_ptr<ttml::autograd::ModuleBase> tok_emb;
    std::vector<std::shared_ptr<ModuleBase>> blocks;
    std::shared_ptr<ModuleBase> ln_fc;
    std::shared_ptr<ttml::autograd::ModuleBase> fc;
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit Qwen(const QwenConfig& config);
    virtual ~Qwen() = default;
    void load_from_safetensors(const std::filesystem::path& model_path) override;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;
};

[[nodiscard]] QwenConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const QwenConfig& qwen_config);
[[nodiscard]] std::shared_ptr<Qwen> create(const QwenConfig& config);
[[nodiscard]] std::shared_ptr<Qwen> create(const YAML::Node& config);

void load_model_from_safetensors(
    const std::filesystem::path& path, serialization::NamedParameters& parameters, const QwenConfig& config);

}  // namespace ttml::models::qwen
