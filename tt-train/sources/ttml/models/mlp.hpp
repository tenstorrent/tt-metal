// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <yaml-cpp/yaml.h>

#include <memory>

#include "modules/multi_layer_perceptron.hpp"

namespace ttml::models::mlp {
[[nodiscard]] ttml::modules::MultiLayerPerceptronParameters read_config(const YAML::Node& config);
[[nodiscard]] YAML::Node write_config(ttml::modules::MultiLayerPerceptronParameters& mlp_config);
[[nodiscard]] std::shared_ptr<ttml::modules::MultiLayerPerceptron> create(
    const ttml::modules::MultiLayerPerceptronParameters& config);
[[nodiscard]] std::shared_ptr<ttml::modules::MultiLayerPerceptron> create(const YAML::Node& config);

}  // namespace ttml::models::mlp
