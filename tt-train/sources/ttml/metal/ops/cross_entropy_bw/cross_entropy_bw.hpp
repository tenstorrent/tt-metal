// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_bw {

struct CrossEntropyBackwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,   // logits : model output (N, 1, H, W)
        const ttnn::Tensor& target,  // target : ground truth (N, H)
        const ttnn::Tensor& grad,    // grad : support only  (1, 1, 1, 1)
        float scaler);
};
}  // namespace ttml::metal::ops::cross_entropy_bw
