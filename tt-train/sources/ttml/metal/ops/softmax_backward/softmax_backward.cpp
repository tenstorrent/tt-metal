// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward.hpp"

namespace ttml::metal {

ttnn::Tensor softmax_backward(
    const ttnn::Tensor& softmax_output,
    const ttnn::Tensor& grad,
    int32_t dim,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    throw std::runtime_error("softmax_backward operation has been removed");
}

}  // namespace ttml::metal
