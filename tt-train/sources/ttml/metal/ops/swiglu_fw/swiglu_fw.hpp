// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_fw {

struct SwiGLUForwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w2, const ttnn::Tensor& w3);
};

}  // namespace ttml::metal::ops::swiglu_fw
