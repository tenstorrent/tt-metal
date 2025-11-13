// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal{

ttnn::Tensor adamw_fused(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg_in,
    const ttnn::Tensor& exp_avg_sq_in,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const uint32_t step);
}  // namespace ttml::metal
