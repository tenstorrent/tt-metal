// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_split.hpp"

#include "device/transpose_split_device_operation.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> transpose_split(
    const ttnn::Tensor& input, uint32_t rows, uint32_t tail_rows, float tail_scale) {
    auto [dx, tail] = ttnn::prim::ttml_transpose_split(input, rows, tail_rows, tail_scale);
    return {dx, tail};
}

}  // namespace ttml::metal
