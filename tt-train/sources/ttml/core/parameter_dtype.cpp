// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/parameter_dtype.hpp"

#include <enchantum/enchantum.hpp>
#include <string>

#include "autograd/auto_context.hpp"

namespace ttml::core {

namespace {

bool is_float_dtype(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::BFLOAT16 || dtype == ttnn::DataType::FLOAT32;
}

// The dtype training_config.precision.<key> names: "bfloat16" or "float32", any other name is a TT_FATAL. Unset, or set
// to null, gives nullopt.
std::optional<ttnn::DataType> precision_dtype_from_config(const YAML::Node& training_config, const char* key) {
    if (!training_config || !training_config["precision"]) {
        return std::nullopt;
    }
    const auto node = training_config["precision"][key];
    if (!node || node.IsNull()) {
        return std::nullopt;
    }
    const auto name = node.as<std::string>();
    if (name == "bfloat16") {
        return ttnn::DataType::BFLOAT16;
    }
    TT_FATAL(name == "float32", "precision.{} must be \"bfloat16\" or \"float32\", got \"{}\"", key, name);
    return ttnn::DataType::FLOAT32;
}

}  // namespace

std::optional<ttnn::DataType> reduce_dtype_from_config(const YAML::Node& training_config) {
    return precision_dtype_from_config(training_config, "reduce_dtype");
}

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
    // Both names are read before anything changes, so a wrong one leaves the parameters and the default as they were.
    const auto param_dtype = precision_dtype_from_config(training_config, "param_dtype");
    const auto reduce_dtype = reduce_dtype_from_config(training_config);
    if (param_dtype) {
        store_trainable_parameters_as(parameters, *param_dtype);
    }
    if (reduce_dtype) {
        autograd::ctx().set_gradient_reduce_dtype(reduce_dtype);
    }
    return param_dtype;
}

}  // namespace ttml::core
