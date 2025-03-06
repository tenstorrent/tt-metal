// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "models/gpt2.hpp"
#include "modules/distributed/gpt_block.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/positional_embeddings.hpp"

namespace ttml::models::distributed::gpt2 {

using models::gpt2::PositionalEmbeddingType;
using models::gpt2::RunnerType;
using models::gpt2::TransformerConfig;
using models::gpt2::WeightTyingType;

class DistributedTransformer : public ttml::autograd::ModuleBase {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::autograd::ModuleBase> tok_emb;
    std::shared_ptr<ttml::autograd::ModuleBase> pos_emb;
    std::vector<std::shared_ptr<ttml::autograd::ModuleBase>> blocks;
    std::shared_ptr<ttml::autograd::ModuleBase> ln_fc;
    std::shared_ptr<ttml::autograd::ModuleBase> fc;

public:
    explicit DistributedTransformer(const TransformerConfig& config);

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;
};

[[nodiscard]] std::shared_ptr<DistributedTransformer> create(const TransformerConfig& config);
[[nodiscard]] std::shared_ptr<DistributedTransformer> create(const YAML::Node& config);

}  // namespace ttml::models::distributed::gpt2
