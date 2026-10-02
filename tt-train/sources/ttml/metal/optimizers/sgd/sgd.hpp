// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::autograd {
class MutableTensorView;
}  // namespace ttml::autograd

namespace ttml::metal {

// Updates param and, when given, momentum_buffer in place. They are taken as views from get_value_for_update() so
// that the derived copies of those tensors are refreshed after the write.

ttnn::Tensor sgd(
    const autograd::MutableTensorView& param,
    const ttnn::Tensor& grad,
    const float lr,
    const float momentum,
    const float dampening,
    const float weight_decay,
    const bool nesterov,
    // nullptr unless momentum is enabled.
    const autograd::MutableTensorView* momentum_buffer);

}  // namespace ttml::metal
