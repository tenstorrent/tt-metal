// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw {

struct CrossEntropyForwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,  // logits : model output (N, 1, H, W)
        const ttnn::Tensor& target_tensor  // target : ground truth (N, H)
    );
};
}  // namespace ttml::metal::ops::cross_entropy_fw
