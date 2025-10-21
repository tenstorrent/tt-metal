// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw.hpp"

#include "device/cross_entropy_fw_device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_fw {

ttnn::Tensor CrossEntropyForwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor, const uint32_t ignore_index) {
    auto result = ttnn::prim::ttml_cross_entropy_fw(input_tensor, target_tensor, ignore_index);
    return result;
}
}  // namespace ttml::metal::ops::cross_entropy_fw
