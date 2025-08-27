// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_fw {

struct SwiGLUForwardOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& m1,
        const ttnn::Tensor& m2,
        const ttnn::Tensor& m3,
        const ttnn::Tensor& m_dropout);
};

}  // namespace ttml::metal::ops::swiglu_fw
