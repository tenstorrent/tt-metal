// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed {
tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b, bool transpose_a = false, bool transpose_b = false);

std::pair<tt::tt_metal::Tensor, tt::tt_metal::Tensor> matmul_backward(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    const tt::tt_metal::Tensor& out_grad,
    bool transpose_a = false,
    bool transpose_b = false);
}  // namespace ttml::ttnn_fixed
