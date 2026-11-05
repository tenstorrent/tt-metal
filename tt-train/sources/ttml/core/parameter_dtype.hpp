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
// Call it after the model is built and before the optimizer is created, so the optimizer state follows the new dtype.
// Loading has to fit around it:
// - Load a checkpoint after it. Checkpoint loading uses assign(), which casts into the dtype a tensor is stored in, so
//   an fp32 checkpoint loaded while the parameters are still bf16 is rounded to bf16, and converting afterwards only
//   widens the rounded values.
// - Run loaders that install tensors with set_value(), such as the C++ safetensors loaders, before it. Run after it,
//   they put bf16 tensors back.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

// Checks training_config.precision without applying it: the block must be a map of known keys (or null), and each
// dtype name "bfloat16" or "float32". Anything else is the TT_FATAL apply_precision_config() would raise. A runner
// calls it right after loading its config, so a typo stops the run before the model is built.
void validate_precision_config(const YAML::Node& training_config);

// Applies training_config.precision. precision.master_dtype ("bfloat16" or "float32") stores the parameters in that
// dtype with store_trainable_parameters_as(), whose contract applies; unset, or set to null, leaves them in the dtype
// the model builds them in. precision.reduce_dtype (same names) becomes the dtype the gradient syncs sum in when a call
// passes none (autograd::AutoContext::set_gradient_reduce_dtype), stored every time: unset or null stores none, so each
// gradient is summed in its own dtype and nothing carries over from a config applied earlier in the process. A script
// that sets the default itself does so after this call. With reduce_dtype "float32" it also sets FLOAT32 as the
// gradient accumulation dtype of every trainable parameter (autograd::Tensor::set_grad_accumulation_dtype), so their
// micro-batch sums run in fp32 too; otherwise it clears that dtype on them. Frozen parameters keep theirs. The block
// must be a map of known keys, and any other key or name is a TT_FATAL; both are read before anything changes. Returns
// the master dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

// Installs a bf16 copy of every fp32 gradient of the parameters, which frees the fp32 one; bf16 gradients and
// parameters without one are left as they are. With reduce_dtype "float32" a trainable parameter's gradient can be fp32
// (see apply_precision_config), and the consumers that take bf16 gradients only call this right before they read them:
// the optimizers other than fused AdamW (which reads an fp32 gradient as it is), clip_grad_norm and RemoteOptimizer
// before it sends. It is the gradient's one rounding to bf16, and calling it again does nothing.
void cast_fp32_gradients_to_bf16(const serialization::NamedParameters& parameters);

}  // namespace ttml::core
