// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <yaml-cpp/yaml.h>

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

struct ClipGradNormConfig {
    float max_norm;
    float p_norm_type;
    bool error_if_nonfinite;
};

// Clip the gradients of the parameters up to a given maximum norm. If
// error_if_nonfinite is true, an error is thrown if the sum of the parameters
// is in {nan,inf,-inf}. p_norm_type specifies which p-norm
// (https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm) to use in the norm
// calculation. Gradients are clipped in place in keeping with pytorch:
// https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html

// Returns the summed norm of the gradients after clipping.
autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type = 2.0F,
    bool error_if_nonfinite = true);

autograd::TensorPtr clip_grad_norm(const serialization::NamedParameters& parameters, const ClipGradNormConfig& config);

}  // namespace ttml::core

// for parsing from yaml model config
namespace YAML {
template <>
struct convert<std::optional<ttml::core::ClipGradNormConfig>> {
    static Node encode(const std::optional<ttml::core::ClipGradNormConfig>& config) {
        if (!config)
            return Node(NodeType::Null);

        Node node;
        node["max_norm"] = config->max_norm;
        node["p_norm_type"] = config->p_norm_type;
        node["error_if_nonfinite"] = config->error_if_nonfinite;
        return node;
    }

    static bool decode(const Node& node, std::optional<ttml::core::ClipGradNormConfig>& config) {
        if (!node || node.IsNull()) {
            config = std::nullopt;
            return true;
        }

        ttml::core::ClipGradNormConfig cfg;
        cfg.max_norm = node["max_norm"].as<float>();
        cfg.p_norm_type = node["p_norm_type"].as<float>(2.0f);
        cfg.error_if_nonfinite = node["error_if_nonfinite"].as<bool>(false);
        config = cfg;
        return true;
    }
};
}  // namespace YAML
