// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_fused.hpp"

#include "device/adamw_device_operation.hpp"

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
    const uint32_t step) {
    return ttnn::prim::ttml_adamw(
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
        stochastic_rounding,
        step);
}

}  // namespace ttml::metal