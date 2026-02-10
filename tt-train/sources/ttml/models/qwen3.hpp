// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include "autograd/tensor.hpp"
#include "base_transformer.hpp"
#include "common/transformer_common.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/qwen3_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::qwen3 {

using RunnerType = common::transformer::RunnerType;
using WeightTyingType = common::transformer::WeightTyingType;

struct Qwen3Config {
    uint32_t num_heads = 16U;
    uint32_t num_groups = 8U;
    uint32_t embedding_dim = 1024U;  // hidden_size
    uint32_t head_dim = 128U;        // ← Explicit head dimension (key difference from Llama)
    std::optional<uint32_t> intermediate_dim = std::nullopt;
    float dropout_prob = 0.0F;
    float theta = 1000000.0F;    // RoPE theta (2x Llama3)
    float rms_norm_eps = 1e-6F;  // Qwen3 uses 1e-6 (vs Llama's 1e-5)
    uint32_t num_blocks = 28U;
    uint32_t vocab_size = 151936U;
    uint32_t max_sequence_length = 2048U;
    RunnerType runner_type = RunnerType::Default;
    WeightTyingType weight_tying = WeightTyingType::Enabled;  // Qwen3 uses weight tying by default

    // RoPE NTK-aware scaling parameters
    float scaling_factor = 0.0F;  // 0.0 means no scaling
    float high_freq_factor = 4.0F;
    float low_freq_factor = 1.0F;
    uint32_t original_context_length = 0U;
};

class Qwen3 : public BaseTransformer {
private:
    RunnerType runner_type = RunnerType::Default;
    Qwen3Config m_config;
    std::shared_ptr<modules::Embedding> tok_emb;
    std::vector<std::shared_ptr<modules::Qwen3Block>> blocks;
    std::shared_ptr<modules::RMSNormLayer> ln_fc;
    std::shared_ptr<modules::LinearLayer> fc;
    ops::RotaryEmbeddingParams m_rope_params;
    uint32_t m_original_vocab_size = 0U;

public:
    explicit Qwen3(const Qwen3Config& config);
    virtual ~Qwen3() = default;
    void load_from_safetensors(const std::filesystem::path& model_path) override;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x,
        const ttml::autograd::TensorPtr& mask,
        std::shared_ptr<common::transformer::KvCache> kv_cache,
        const uint32_t new_tokens);

    // Get the original vocabulary size for token validation
    [[nodiscard]] uint32_t get_original_vocab_size() const {
        return m_original_vocab_size;
    }

    [[nodiscard]] std::shared_ptr<modules::Qwen3Block> get_block(size_t index) const {
        if (index >= blocks.size()) {
            throw std::out_of_range(fmt::format("Block index {} out of range (max: {})", index, blocks.size() - 1));
        }
        return blocks[index];
    }

    // Get number of blocks
    [[nodiscard]] size_t num_blocks() const {
        return blocks.size();
    }
};

[[nodiscard]] Qwen3Config read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(const Qwen3Config& qwen3_config);
[[nodiscard]] std::shared_ptr<Qwen3> create(const Qwen3Config& config);
[[nodiscard]] std::shared_ptr<Qwen3> create(const YAML::Node& config);

void load_model_from_safetensors(
    const std::filesystem::path& path,
    serialization::NamedParameters& parameters,
    const Qwen3Config& config,
    std::set<std::string>& used_parameters,
    std::set<std::string>& ignored_parameters,
    bool verbose = false);

}  // namespace ttml::models::qwen3
