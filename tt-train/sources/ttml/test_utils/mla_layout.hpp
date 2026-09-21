// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::test_utils {

// packed [B, 1, S, H*D] -> head-major [B, H, S, D]
inline ttnn::Tensor packed_to_head_major(const ttnn::Tensor& packed, uint32_t n_heads, uint32_t head_dim) {
    const auto shape = packed.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t S = shape[2];
    return ttnn::transpose(ttnn::reshape(packed, ttnn::Shape({B, S, n_heads, head_dim})), 1, 2);
}

// head-major [B, H, S, D] -> packed [B, 1, S, H*D]
inline ttnn::Tensor head_major_to_packed(const ttnn::Tensor& head_major) {
    const auto shape = head_major.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t H = shape[1];
    const uint32_t S = shape[2];
    const uint32_t D = shape[3];
    return ttnn::reshape(ttnn::transpose(head_major, 1, 2), ttnn::Shape({B, 1U, S, H * D}));
}

}  // namespace ttml::test_utils
