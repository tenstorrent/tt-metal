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

// Applies training_config.precision. precision.param_dtype ("bfloat16" or "float32"; any other name is a TT_FATAL)
// stores the parameters in that dtype; unset, or set to null, leaves them in the dtype the model builds them in.
// precision.reduce_dtype (same names) becomes the dtype the gradient syncs sum in when a call passes none
// (autograd::AutoContext::set_gradient_reduce_dtype); unset or null leaves that default as it is.
// Called right before the optimizer is created by ttml.common.utils.create_optimizer() (given the whole config), the
// nano_gpt runner and the 3-tier optimizer worker. Returns the param dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

// The dtype training_config.precision.reduce_dtype names ("bfloat16" or "float32"; any other name is a TT_FATAL), or
// nullopt when it is unset or null. FSDP takes it as an argument of fully_shard(), which runs before the optimizer is
// created, so train.py reads it with this function too.
std::optional<ttnn::DataType> reduce_dtype_from_config(const YAML::Node& training_config);

}  // namespace ttml::core
