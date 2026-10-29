// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <optional>

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

// Stores every trainable bf16 or fp32 parameter in dtype (BFLOAT16 or FLOAT32), cast from its current value. Frozen
// parameters and non-float tensors keep their dtype. Call it after the model is built and before the optimizer is
// created, so the optimizer state follows the new dtype. With FLOAT32, fused AdamW keeps the fp32 master weights in
// the parameters themselves, and forward and backward run on their bf16 view.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

// Applies training_config.precision to the parameters: stores them in precision.param_dtype ("bfloat16" or "float32";
// any other name is a TT_FATAL). Unset, or set to null, leaves the parameters in the dtype the model builds them in.
// Called right before the optimizer is created by ttml.common.utils.create_optimizer() (given the whole config), the
// nano_gpt runner and the 3-tier optimizer worker. Returns the dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

}  // namespace ttml::core
