// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/parameter_dtype.hpp"

#include <enchantum/enchantum.hpp>
#include <string>
#include <string_view>

namespace ttml::core {

namespace {

bool is_float_dtype(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::BFLOAT16 || dtype == ttnn::DataType::FLOAT32;
}

ttnn::DataType parse_param_dtype(std::string_view name) {
    if (name == "bfloat16") {
        return ttnn::DataType::BFLOAT16;
    }
    TT_FATAL(name == "float32", "precision.param_dtype must be \"bfloat16\" or \"float32\", got \"{}\"", name);
    return ttnn::DataType::FLOAT32;
}

std::optional<ttnn::DataType> param_dtype_from_config(const YAML::Node& training_config) {
    if (!training_config || !training_config["precision"]) {
        return std::nullopt;
    }
    const auto name = training_config["precision"]["param_dtype"];
    if (!name || name.IsNull()) {
        return std::nullopt;
    }
    return parse_param_dtype(name.as<std::string>());
}

}  // namespace

void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype) {
    TT_FATAL(
        is_float_dtype(dtype), "Parameters can be stored as BFLOAT16 or FLOAT32, got {}", enchantum::to_string(dtype));
    const auto precision =
        dtype == ttnn::DataType::FLOAT32 ? autograd::PreferredPrecision::FULL : autograd::PreferredPrecision::HALF;
    for (const auto& [name, parameter] : parameters) {
        if (!parameter->get_requires_grad()) {
            continue;
        }
        const auto native_dtype = parameter->get_value(autograd::PreferredPrecision::NATIVE).dtype();
        if (!is_float_dtype(native_dtype) || native_dtype == dtype) {
            continue;
        }
        // The view of the other precision is the cast value; installing it makes it the native tensor.
        parameter->set_value(parameter->get_value(precision));
    }
}

std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters) {
    const auto param_dtype = param_dtype_from_config(training_config);
    if (param_dtype) {
        store_trainable_parameters_as(parameters, *param_dtype);
    }
    return param_dtype;
}

}  // namespace ttml::core
