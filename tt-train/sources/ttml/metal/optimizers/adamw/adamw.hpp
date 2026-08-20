// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor adamw(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    const std::optional<ttnn::Tensor>& max_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float beta1_pow,
    float beta2_pow,
    float epsilon,
    float weight_decay,
    StochasticRounding stochastic_rounding = StochasticRounding::Disabled,
    // Required iff stochastic rounding is enabled.
    std::optional<uint32_t> stochastic_rounding_seed = std::nullopt);

// AdamW with the step-varying scalars as single-element f32 device tensors.
//   step_size    = lr / (1 - beta1^t)
//   inv_sqrt_bc2 = 1 / sqrt(1 - beta2^t)
//   decay_factor = 1 - lr * weight_decay
//
// Stochastic rounding is deliberately not offered on this overload. It needs a
// fresh host-drawn seed every step, delivered as a compute runtime argument --
// exactly the per-step host work this overload exists to remove. Use the
// float-scalar `adamw` above when stochastic rounding is required.
ttnn::Tensor adamw_tensor_scalars(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    const std::optional<ttnn::Tensor>& max_exp_avg_sq,
    const ttnn::Tensor& step_size,
    const ttnn::Tensor& inv_sqrt_bc2,
    const ttnn::Tensor& decay_factor,
    float beta1,
    float beta2,
    float epsilon);

}  // namespace ttml::metal
