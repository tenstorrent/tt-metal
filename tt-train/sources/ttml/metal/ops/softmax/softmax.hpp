// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::softmax {

struct SoftmaxOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input, int32_t dim);
};
}  // namespace ttml::metal::ops::softmax
