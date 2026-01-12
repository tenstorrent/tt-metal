// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "models/llama.hpp"
#include "modules/llama_block.hpp"
#include "modules/module_base.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::distributed::llama {

using RunnerType = common::transformer::RunnerType;
using models::llama::LlamaConfig;

class DistributedLlama : public BaseTransformer {
private:
    RunnerType runner_type = RunnerType::Default;
    // Use ModuleBasePtr to allow replacement with LoRA layers
    ttml::modules::ModuleBasePtr tok_emb;
    std::vector<ttml::modules::ModuleBasePtr> blocks;
    ttml::modules::ModuleBasePtr ln_fc;
    ttml::modules::ModuleBasePtr fc;
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit DistributedLlama(const LlamaConfig& config);
    virtual ~DistributedLlama() = default;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;
};

[[nodiscard]] std::shared_ptr<DistributedLlama> create(const LlamaConfig& config);
[[nodiscard]] std::shared_ptr<DistributedLlama> create(const YAML::Node& config);

}  // namespace ttml::models::distributed::llama
