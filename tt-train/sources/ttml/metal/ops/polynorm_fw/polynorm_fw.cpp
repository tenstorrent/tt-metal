// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw.hpp"

#include "device/polynorm_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm3_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& weight, const ttnn::Tensor& bias, float epsilon) {
    return ttnn::prim::ttml_polynorm3_fw(input_tensor, weight, bias, epsilon);
}

}  // namespace ttml::metal
