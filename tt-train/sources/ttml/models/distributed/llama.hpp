// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "models/common/transformer_common.hpp"
#include "models/llama.hpp"
#include "modules/llama_block.hpp"
#include "ops/rope_op.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::distributed::llama {

using RunnerType = common::transformer::RunnerType;
using models::llama::LlamaConfig;

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

[[nodiscard]] std::shared_ptr<DistributedLlama> create(const LlamaConfig& config);
[[nodiscard]] std::shared_ptr<DistributedLlama> create(const YAML::Node& config);

}  // namespace ttml::models::distributed::llama
