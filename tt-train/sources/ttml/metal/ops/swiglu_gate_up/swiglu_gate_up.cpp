// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_gate_up.hpp"

#include "device/swiglu_gate_up_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_gate_up(const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w3) {
    return ttnn::prim::ttml_swiglu_gate_up(input_tensor, w1, w3);
}

}  // namespace ttml::metal
