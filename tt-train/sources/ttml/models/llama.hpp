// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "base_transformer.hpp"
#include "common/transformer_common.hpp"
#include "modules/llama_block.hpp"
#include "modules/module_base.hpp"
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
    WeightTyingType weight_tying = WeightTyingType::Enabled;

    // RoPE NTK-aware scaling parameters
    float scaling_factor = 0.0F;  // 0.0 means no scaling
    float high_freq_factor = 4.0F;
    float low_freq_factor = 1.0F;
    uint32_t original_context_length = 0U;
};

class Llama : public BaseTransformer {
private:
    RunnerType runner_type = RunnerType::Default;
    LlamaConfig m_config;
    std::shared_ptr<ttml::modules::ModuleBase> tok_emb;
    std::vector<std::shared_ptr<ModuleBase>> blocks;
    std::shared_ptr<ModuleBase> ln_fc;
    std::shared_ptr<ttml::modules::ModuleBase> fc;
    ops::RotaryEmbeddingParams m_rope_params;
    uint32_t m_original_vocab_size = 0U;

public:
    explicit Llama(const LlamaConfig& config);
    virtual ~Llama() = default;
    void load_from_safetensors(const std::filesystem::path& model_path) override;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;

    // Get the original vocabulary size for token validation
    [[nodiscard]] uint32_t get_original_vocab_size() const {
        return m_original_vocab_size;
    }
};

[[nodiscard]] LlamaConfig read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const LlamaConfig& llama_config);
[[nodiscard]] std::shared_ptr<Llama> create(const LlamaConfig& config);
[[nodiscard]] std::shared_ptr<Llama> create(const YAML::Node& config);

void load_model_from_safetensors(
    const std::filesystem::path& path, serialization::NamedParameters& parameters, const LlamaConfig& config);
}  // namespace ttml::models::llama
