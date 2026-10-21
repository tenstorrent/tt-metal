// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

// The dtype named by training_config.precision.param_dtype: "bfloat16" or "float32". Any other name is a TT_FATAL.
[[nodiscard]] ttnn::DataType parse_param_dtype(std::string_view name);

// Stores every trainable bf16 or fp32 parameter in dtype (BFLOAT16 or FLOAT32), cast from its current value. Frozen
// parameters and non-float tensors keep their dtype. Call it after the model is built and before the optimizer is
// created, so the optimizer state follows the new dtype. With FLOAT32, fused AdamW keeps the fp32 master weights in
// the parameters themselves, and forward and backward run on their bf16 view.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

}  // namespace ttml::core
