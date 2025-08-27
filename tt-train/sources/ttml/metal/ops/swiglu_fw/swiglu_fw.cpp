// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw.hpp"

#include <iostream>

#include "core/compute_kernel_config.hpp"
#include "device/swiglu_fw_device_operation.hpp"
#include "metal/ops/swiglu_fw/swiglu_fw.hpp"

namespace ttml::metal::ops::swiglu_fw {

ttnn::Tensor SwiGLUForwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w2, const ttnn::Tensor& w3) {
    return ttnn::prim::ttml_swiglu_fw(
        input_tensor,  // [B,1,S,C]
        w1,            // [C,H]
        w2,            // [H,C]
        w3             // [C,H]
    );
}

}  // namespace ttml::metal::ops::swiglu_fw
