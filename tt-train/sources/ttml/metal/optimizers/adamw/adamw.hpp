// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::autograd {
class MutableTensorView;
}  // namespace ttml::autograd

namespace ttml::metal {

// Updates param, exp_avg, exp_avg_sq and, when given, max_exp_avg_sq in place. They are taken as views from
// get_value_for_update() so that the derived copies of those tensors are refreshed after the write.

ttnn::Tensor adamw(
    const autograd::MutableTensorView& param,
    const ttnn::Tensor& grad,
    const autograd::MutableTensorView& exp_avg,
    const autograd::MutableTensorView& exp_avg_sq,
    // nullptr unless amsgrad is enabled.
    const autograd::MutableTensorView* max_exp_avg_sq,
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

}  // namespace ttml::metal
