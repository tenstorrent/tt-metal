// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw.hpp"

#include "autograd/autocast_tensor.hpp"
#include "device/adamw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor adamw(
    const autograd::MutableTensorView& param,
    const ttnn::Tensor& grad,
    const autograd::MutableTensorView& exp_avg,
    const autograd::MutableTensorView& exp_avg_sq,
    const autograd::MutableTensorView* max_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float beta1_pow,
    float beta2_pow,
    float epsilon,
    float weight_decay,
    StochasticRounding stochastic_rounding,
    std::optional<uint32_t> stochastic_rounding_seed) {
    TT_FATAL(
        (stochastic_rounding == StochasticRounding::Enabled) == stochastic_rounding_seed.has_value(),
        "a stochastic rounding seed must be supplied iff stochastic rounding is enabled");
    const auto max_exp_avg_sq_tensor =
        max_exp_avg_sq != nullptr ? std::optional<ttnn::Tensor>(max_exp_avg_sq->tensor()) : std::nullopt;
    return ttnn::prim::adamw(
        param.tensor(),
        grad,
        exp_avg.tensor(),
        exp_avg_sq.tensor(),
        max_exp_avg_sq_tensor,
        lr,
        beta1,
        beta2,
        beta1_pow,
        beta2_pow,
        epsilon,
        weight_decay,
        max_exp_avg_sq != nullptr,
        stochastic_rounding,
        stochastic_rounding_seed);
}

}  // namespace ttml::metal
