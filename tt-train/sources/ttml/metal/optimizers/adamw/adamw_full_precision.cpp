// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_full_precision.hpp"

#include "device/adamw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor adamw_full_precision(
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
    float weight_decay) {
    return ttnn::prim::adamw(
        param_in,
        grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        beta1_pow,
        beta2_pow,
        epsilon,
        weight_decay,
        max_exp_avg_sq.has_value(),
        false);  // stochastic_rounding disabled for full precision
}

}  // namespace ttml::metal
