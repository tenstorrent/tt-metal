// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor cross_entropy_bw(
    const ttnn::Tensor& input,   // logits : model output (N, 1, H, W)
    const ttnn::Tensor& target,  // target : ground truth (N, H)
    const ttnn::Tensor& grad,    // grad : support only  (1, 1, 1, 1)
    float scaler);

}  // namespace ttml::metal
