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
// Call it after the model is built and before any values are loaded and the optimizer is created, so the loaded values
// and the optimizer state follow the new dtype. Checkpoint loading and the C++ safetensors loaders install values with
// assign() (serialization::assign_from_floats() for the latter), which keeps the dtype a tensor is stored in: loaded
// after the conversion, an fp32 source keeps its bits in fp32 parameters; loaded before it, it is rounded to bf16, and
// converting afterwards only widens the rounded values.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

// Checks training_config.precision without applying it: the block must be a map of known keys (or null), and each
// dtype name "bfloat16" or "float32". Anything else is the TT_FATAL apply_precision_config() would raise. The runners'
// config parsers call it, so a typo stops the run before the model is built.
void validate_precision_config(const YAML::Node& training_config);

// Applies training_config.precision. precision.master_dtype ("bfloat16" or "float32") stores the parameters in that
// dtype with store_trainable_parameters_as(), whose contract applies; unset, or set to null, leaves them in the dtype
// the model builds them in. precision.reduce_dtype (same names) is applied as apply_reduce_dtype_config() does. The
// block must be a map of known keys, and any other key or name is a TT_FATAL; both are read before anything changes.
// Returns the master dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

// Applies only training_config.precision.reduce_dtype. It becomes the dtype the gradient syncs sum in when a call
// passes none (autograd::AutoContext::set_gradient_reduce_dtype), stored every time: unset or null stores none, so each
// gradient is summed in its own dtype and nothing carries over from a config applied earlier in the process; a script
// that sets the default itself does so after this call. With "float32" every trainable parameter also accumulates its
// gradient in fp32 (autograd::Tensor::set_accumulate_grad_in_fp32), so its micro-batch sums run in fp32 too; otherwise
// that is cleared on them. Frozen parameters are left alone, and no parameter changes its stored dtype, whatever
// master_dtype says, though the whole block is still validated. Used on its own by a process whose parameters stay bf16
// while the master weights live elsewhere, such as the 3-tier workers and aggregator. Returns the reduce dtype it
// stored, if any.
std::optional<ttnn::DataType> apply_reduce_dtype_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

// Installs a bf16 copy of every fp32 gradient of the parameters, which frees the fp32 one; bf16 gradients and
// parameters without one are left as they are. With reduce_dtype "float32" a trainable parameter's gradient can be
// fp32, and whatever reads gradients as bf16 only calls this right before it reads them (see
// optimizers::OptimizerBase::step). It is the gradient's one rounding to bf16, and calling it again does nothing.
void cast_fp32_gradients_to_bf16(const serialization::NamedParameters& parameters);

}  // namespace ttml::core
