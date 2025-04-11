// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw.hpp"

#include "device/cross_entropy_fw_device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_fw {

ttnn::Tensor CrossEntropyForwardOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor) {
    auto result /*ttnn::Tensor*/ = ttnn::prim::ttml_cross_entropy_fw(
        input_tensor, target_tensor);  // logits : model output (B, 1, S, D), target : ground truth (B, 1, S, D)
    // resulut shape (B, 1, S, 1) - will change this later
    return result;
}
}  // namespace ttml::metal::ops::cross_entropy_fw
