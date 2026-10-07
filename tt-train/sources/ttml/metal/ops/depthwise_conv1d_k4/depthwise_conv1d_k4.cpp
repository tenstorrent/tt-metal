// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_k4.hpp"

#include "device/depthwise_conv1d_k4_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor depthwise_conv1d_k4(
    const ttnn::Tensor& input,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    bool anti_causal,
    const std::optional<ttnn::Tensor>& silu_grad) {
    return ttnn::prim::ttml_depthwise_conv1d_k4(input, tap0, tap1, tap2, tap3, anti_causal, silu_grad);
}

}  // namespace ttml::metal
