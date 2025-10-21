// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw {

struct CrossEntropyForwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,   // logits : model output (N, 1, H, W)
        const ttnn::Tensor& target,  // target : ground truth (N, H)
        const uint32_t ignore_index = 1000000000U);
};
}  // namespace ttml::metal::ops::cross_entropy_fw
