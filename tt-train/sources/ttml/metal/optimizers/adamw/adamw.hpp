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
    std::optional<uint32_t> stochastic_rounding_seed = std::nullopt,
    // Iteration 45: the bias-corrected step scalars from a [1, 1, 32, 64] fp32 device tensor (tile 0 the inverse step
    // size (1 - beta1^t) / lr, tile 1 1 / sqrt(1 - beta2^t)) instead of beta1_pow / beta2_pow: a captured step replays
    // with advancing powers.
    const std::optional<ttnn::Tensor>& step_scalars = std::nullopt);

}  // namespace ttml::metal
