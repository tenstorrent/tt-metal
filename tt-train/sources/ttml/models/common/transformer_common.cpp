// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_common.hpp"

#include "yaml-cpp/yaml.h"

namespace ttml::models::common::transformer {

void initialize_weights_gpt2(autograd::ModuleBase& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            init::normal_init(tensor_ptr, tensor.logical_shape(), {0.F, 0.02F});
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.logical_shape(), 0.F);
        }
    }
}

void initialize_weights_he_kaiming_normal(autograd::ModuleBase& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            auto mean = 0.0F;
            // take penultimate dimension as the input dim.
            auto fan_in = tensor.logical_shape()[-2];
            auto stddev = std::sqrt(2.0F / fan_in);
            init::normal_init(tensor_ptr, tensor.logical_shape(), {mean, stddev});
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.logical_shape(), 0.F);
        }
    }
}

RunnerType read_runner_type(const YAML::Node& config) {
    auto runner_type_str = config["runner_type"].as<std::string>("default");
    if (runner_type_str == "default") {
        return RunnerType::Default;
    } else if (runner_type_str == "memory_efficient") {
        return RunnerType::MemoryEfficient;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown runner type: {}. Supported runner types [default, memory_efficient]", runner_type_str));
    }
}

WeightTyingType read_weight_tying_type(const YAML::Node& config) {
    auto weight_tying_str = config["weight_tying"].as<std::string>("disabled");
    if (weight_tying_str == "disabled") {
        return WeightTyingType::Disabled;
    } else if (weight_tying_str == "enabled") {
        return WeightTyingType::Enabled;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown weight tying type: {}. Supported weight tying types [disabled, enabled]", weight_tying_str));
    }
}

}  // namespace ttml::models::common::transformer
