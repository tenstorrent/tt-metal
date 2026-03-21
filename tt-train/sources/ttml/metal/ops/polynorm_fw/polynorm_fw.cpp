// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw.hpp"

#include "device/polynorm_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm_fw(const ttnn::Tensor& input_tensor, float w0, float w1, float w2, float bias, float epsilon) {
    return ttnn::prim::ttml_polynorm_fw(input_tensor, w0, w1, w2, bias, epsilon);
}

}  // namespace ttml::metal
