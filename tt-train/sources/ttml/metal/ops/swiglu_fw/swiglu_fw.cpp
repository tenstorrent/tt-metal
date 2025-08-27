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
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    const ttnn::Tensor& w3,
    const ttnn::Tensor& dropout) {
    std::cerr << "Input tensor: " << std::endl;
    input_tensor.print();
    std::cerr << "w1: " << std::endl;
    w1.print();
    std::cerr << "w2: " << std::endl;
    w2.print();
    // std::cerr << "w3: " << std::endl;
    // w3.print();
    return ttnn::prim::ttml_swiglu_fw(
        // TODO: Shapes to be checked
        input_tensor,  // [B,1,S,C]
        w1,            // [B,1,S,C]
        w2,            // [B,1,S,C]
        w3,            // [B,1,S,C]
        dropout        // [B,1,S,C]
    );
}

}  // namespace ttml::metal::ops::swiglu_fw
