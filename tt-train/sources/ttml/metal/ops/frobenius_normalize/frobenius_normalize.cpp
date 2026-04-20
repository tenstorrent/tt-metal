// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "frobenius_normalize.hpp"

#include "device/frobenius_normalize_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor frobenius_normalize(const ttnn::Tensor& input_tensor, float epsilon) {
    auto result = ttnn::prim::ttml_frobenius_normalize(input_tensor, epsilon);
    return result[0];
}

}  // namespace ttml::metal
