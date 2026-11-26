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
// parameters and non-float tensors keep their dtype. With FLOAT32, fused AdamW keeps the fp32 master weights in the
// parameters themselves, and forward and backward run on their bf16 view.
//
// Load order: build the model, convert, load weights, create the optimizer, then load a training checkpoint, which
// restores the optimizer state too. Every loader (checkpoints through assign(), the C++ safetensors loaders through
// serialization::assign_from_floats()) writes the dtype a tensor is stored in, so a value loaded after the conversion
// keeps its fp32 bits, while one loaded before it is rounded to bf16 and the conversion only widens it. The optimizer
// state follows the dtype the parameters have when it is created.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

// Checks training_config.precision without applying it, with the TT_FATAL apply_precision_config() would raise. The
// runners' config parsers call it, so a typo stops the run before the model is built.
void validate_precision_config(const YAML::Node& training_config);

// Applies training_config.precision. precision.master_dtype ("bfloat16" or "float32") stores the parameters in that
// dtype with store_trainable_parameters_as(), whose contract applies; unset, or set to null, leaves them in the dtype
// the model builds them in. precision.reduce_dtype (same names) becomes the dtype the gradient syncs sum in when a call
// passes none (autograd::AutoContext::set_gradient_reduce_dtype), stored every time: unset or null stores none, so each
// gradient is summed in its own dtype and nothing carries over from a config applied earlier in the process. With
// FLOAT32 stored, module parameters also add up their gradients in fp32 (see autograd::Tensor::add_grad). A script
// that sets the default itself does so after this call. The block must be a map of known keys, and any other key or
// name is a TT_FATAL; both are read before anything changes. Returns the master dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

// Replaces every fp32 gradient of `parameters` with its bf16 cast, freeing the fp32 buffer; bf16 and missing
// gradients are left as they are, so a second call does nothing. Fused AdamW reads fp32 gradients as they are; every
// other consumer that takes bf16 gradients calls this first, so the gradient is rounded to bf16 once.
void cast_fp32_gradients_to_bf16(const serialization::NamedParameters& parameters);

}  // namespace ttml::core
