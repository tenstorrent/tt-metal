// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw {

struct CrossEntropyForwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,  // logits : model output (B, 1, S, D)
        const ttnn::Tensor& target_tensor  // target : ground truth (B, 1, S, D)
    );
};
}  // namespace ttml::metal::ops::cross_entropy_fw
