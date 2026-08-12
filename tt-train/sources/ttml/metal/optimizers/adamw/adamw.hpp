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

// Overload taking the bias-correction terms beta1^t / beta2^t as single-element
// float32 tensors instead of floats. The kernel derives `step_size` and
// `1 / bias_correction2` on device, so a caller that keeps its step counter on
// device never reads beta^t back to host to run an optimizer step.
//
// Both tensors must be FLOAT32. bfloat16 is rejected rather than supported: the
// kernel needs 1 - beta^t, beta^t sits just below 1 for most of a run, and the
// subtraction cancels - bfloat16 rounds beta2^1 = 0.999 to exactly 1.0, which
// would divide by zero. Convert on device before calling if your betas are held
// in lower precision.
ttnn::Tensor adamw(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    const std::optional<ttnn::Tensor>& max_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    const ttnn::Tensor& beta1_pow,
    const ttnn::Tensor& beta2_pow,
    float epsilon,
    float weight_decay,
    StochasticRounding stochastic_rounding = StochasticRounding::Disabled,
    // Required iff stochastic rounding is enabled.
    std::optional<uint32_t> stochastic_rounding_seed = std::nullopt);

}  // namespace ttml::metal
