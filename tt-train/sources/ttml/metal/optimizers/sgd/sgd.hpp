// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor sgd(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    const float lr,
    const float momentum,
    const float dampening,
    const float weight_decay,
    const bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer);

}  // namespace ttml::metal
