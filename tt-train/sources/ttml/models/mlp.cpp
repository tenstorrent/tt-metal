// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlp.hpp"

namespace ttml::models::mlp {

ttml::modules::MultiLayerPerceptronParameters read_config(const YAML::Node& config) {
    ttml::modules::MultiLayerPerceptronParameters mlp_config;
    mlp_config.input_features = config["input_features"].as<uint32_t>();
    mlp_config.hidden_features = config["hidden_features"].as<std::vector<uint32_t>>();
    mlp_config.output_features = config["output_features"].as<uint32_t>();
    return mlp_config;
}

YAML::Node write_config(ttml::modules::MultiLayerPerceptronParameters& mlp_config) {
    YAML::Node config;
    config["input_features"] = mlp_config.input_features;
    config["hidden_features"] = mlp_config.hidden_features;
    config["output_features"] = mlp_config.output_features;
    return config;
}

std::shared_ptr<ttml::modules::MultiLayerPerceptron> create(
    const ttml::modules::MultiLayerPerceptronParameters& config) {
    return std::make_shared<ttml::modules::MultiLayerPerceptron>(config);
}

std::shared_ptr<ttml::modules::MultiLayerPerceptron> create(const YAML::Node& config) {
    ttml::modules::MultiLayerPerceptronParameters mlp_config = read_config(config);
    return std::make_shared<ttml::modules::MultiLayerPerceptron>(mlp_config);
}
}  // namespace ttml::models::mlp
