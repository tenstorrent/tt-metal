// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <optional>

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

class DistributedTransformer : public BaseTransformer {
private:
    RunnerType runner_type = RunnerType::Default;
    std::shared_ptr<ttml::modules::ModuleBase> tok_emb;
    std::shared_ptr<ttml::modules::ModuleBase> pos_emb;
    std::vector<std::shared_ptr<ttml::modules::distributed::DistributedGPTBlock>> blocks;
    std::shared_ptr<ttml::modules::ModuleBase> ln_fc;
    std::shared_ptr<ttml::modules::ModuleBase> fc;

public:
    // `gather_output_at_lm_head` controls whether the column-parallel LM head all-gathers
    // its output to produce fully-replicated [B,1,S,V] logits.  Set to `false` when the
    // downstream loss expects vocab-sharded logits (e.g. ttml::ops::distributed::
    // vocab_parallel_cross_entropy_loss); set to `true` (default) when the downstream loss
    // expects replicated full-vocab logits (e.g. ttml::ops::cross_entropy_loss).
    explicit DistributedTransformer(const TransformerConfig& config, bool gather_output_at_lm_head = true);
    virtual ~DistributedTransformer() = default;
    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const std::optional<ttml::autograd::TensorPtr>& mask) override;
};

[[nodiscard]] std::shared_ptr<DistributedTransformer> create(
    const TransformerConfig& config, bool gather_output_at_lm_head = true);
[[nodiscard]] std::shared_ptr<DistributedTransformer> create(
    const YAML::Node& config, bool gather_output_at_lm_head = true);

}  // namespace ttml::models::distributed::gpt2
