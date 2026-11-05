// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/parameter_dtype.hpp"

#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <enchantum/enchantum.hpp>
#include <string>
#include <string_view>

#include "autograd/auto_context.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::core {

namespace {

bool is_float_dtype(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::BFLOAT16 || dtype == ttnn::DataType::FLOAT32;
}

// The keys training_config.precision may hold.
constexpr std::array<std::string_view, 2> kPrecisionKeys = {"master_dtype", "reduce_dtype"};

// The block is a map of known keys, so a typo or a key this build doesn't know stops the run instead of being ignored.
void validate_precision_block(const YAML::Node& precision) {
    TT_FATAL(
        precision.IsMap() || precision.IsNull(),
        "training_config.precision must be a map of keys such as master_dtype");
    const auto supported = fmt::format("{}", fmt::join(kPrecisionKeys, ", "));
    for (const auto& entry : precision) {
        const auto key = entry.first.as<std::string>();
        TT_FATAL(
            std::ranges::find(kPrecisionKeys, key) != kPrecisionKeys.end(),
            "Unknown key precision.{} (supported: {})",
            key,
            supported);
    }
}

// The dtype precision.<key> names in a validated block: "bfloat16" or "float32", any other name is a TT_FATAL. Unset,
// or set to null, gives nullopt.
std::optional<ttnn::DataType> dtype_from_precision_block(const YAML::Node& precision, std::string_view key) {
    const auto node = precision[std::string(key)];
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

struct PrecisionDtypes {
    std::optional<ttnn::DataType> master_dtype;
    std::optional<ttnn::DataType> reduce_dtype;
};

// training_config.precision read once: the block is validated, then both keys are read, before any caller changes
// anything.
PrecisionDtypes read_precision_config(const YAML::Node& training_config) {
    if (!training_config || !training_config["precision"]) {
        return {};
    }
    const auto precision = training_config["precision"];
    validate_precision_block(precision);
    auto master_dtype = dtype_from_precision_block(precision, "master_dtype");
    auto reduce_dtype = dtype_from_precision_block(precision, "reduce_dtype");
    return {master_dtype, reduce_dtype};
}

// Stores reduce_dtype as the gradient syncs' default and sets FLOAT32 gradient accumulation on the trainable parameters
// when it is FLOAT32, clearing it otherwise.
void apply_reduce_dtype(std::optional<ttnn::DataType> reduce_dtype, const serialization::NamedParameters& parameters) {
    autograd::ctx().set_gradient_reduce_dtype(reduce_dtype);
    const auto accumulation_dtype =
        reduce_dtype == ttnn::DataType::FLOAT32 ? std::optional{ttnn::DataType::FLOAT32} : std::nullopt;
    for (const auto& [name, parameter] : parameters) {
        if (parameter->get_requires_grad()) {
            parameter->set_grad_accumulation_dtype(accumulation_dtype);
        }
    }
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

void validate_precision_config(const YAML::Node& training_config) {
    (void)read_precision_config(training_config);
}

std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters) {
    const auto dtypes = read_precision_config(training_config);
    if (dtypes.master_dtype) {
        store_trainable_parameters_as(parameters, *dtypes.master_dtype);
    }
    apply_reduce_dtype(dtypes.reduce_dtype, parameters);
    return dtypes.master_dtype;
}

std::optional<ttnn::DataType> apply_reduce_dtype_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters) {
    const auto dtypes = read_precision_config(training_config);
    apply_reduce_dtype(dtypes.reduce_dtype, parameters);
    return dtypes.reduce_dtype;
}

void cast_fp32_gradients_to_bf16(const serialization::NamedParameters& parameters) {
    for (const auto& [name, parameter] : parameters) {
        if (!parameter->is_grad_initialized()) {
            continue;
        }
        const auto& grad = parameter->get_grad();
        if (grad.dtype() == ttnn::DataType::FLOAT32) {
            parameter->set_grad(ttnn::typecast(grad, ttnn::DataType::BFLOAT16));
        }
    }
}

}  // namespace ttml::core
