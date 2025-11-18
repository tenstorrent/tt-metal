// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/relu_sq_bw_device_operation.hpp"  // Includes the ttnn::prim registration
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::relu_sq_bw {

struct ReLUSquaredBackwardOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor);
};

}  // namespace ttml::metal::ops::relu_sq_bw
