// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor adamw_fused(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    const std::optional<ttnn::Tensor>& max_exp_avg_sq,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_pow,
    const float beta2_pow,
    const float epsilon,
    const float weight_decay,
    const bool stochastic_rounding,
    const uint32_t step);

}  // namespace ttml::metal
