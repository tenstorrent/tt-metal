// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw.hpp"

#include "device/cross_entropy_bw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor cross_entropy_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor, const ttnn::Tensor& grad, float scaler) {
    // Subtract the one-hot target before applying the reduction scale. The primitive's output
    // crosses a BF16 circular-buffer boundary before its writer updates the target lane, so passing
    // a non-unit scaler into the primitive computes BF16(BF16(s * p) - s) instead of one final
    // quantization of s * (p - 1).
    auto output = ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, /* scaler */ 1.0F);
    if (scaler == 1.0F) {
        return ttnn::multiply(output, grad);
    }

    // Scale the scalar/per-position upstream gradient instead of the full logits-shaped output.
    const auto scaled_grad = ttnn::multiply(grad, scaler);
    return ttnn::multiply(output, scaled_grad);
}

}  // namespace ttml::metal
