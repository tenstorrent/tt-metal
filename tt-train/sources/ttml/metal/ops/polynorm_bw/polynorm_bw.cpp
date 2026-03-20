// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw.hpp"

#include "device/polynorm_bw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor, float w0, float w1, float w2, float epsilon) {
    return ttnn::prim::ttml_polynorm_bw(input_tensor, dL_dout_tensor, w0, w1, w2, epsilon);
}

}  // namespace ttml::metal
