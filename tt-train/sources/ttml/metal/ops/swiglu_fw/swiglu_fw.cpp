// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw.hpp"

#include "device/swiglu_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w2, const ttnn::Tensor& w3) {
    return ttnn::prim::ttml_swiglu_fw(input_tensor, w1, w2, w3, std::nullopt);
}

}  // namespace ttml::metal
