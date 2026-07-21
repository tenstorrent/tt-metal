// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_q_rope::device {

struct MlaQRopeParams {
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    // true:  packed [B,1,S,H*D] -> head-major [B,H,S,D]  (forward)
    // false: head-major [B,H,S,D] -> packed [B,1,S,H*D]  (backward)
    bool packed_input = true;
};

struct MlaQRopeInputs {
    const ttnn::Tensor& q_in;
    const ttnn::Tensor& cos_cache;
    const ttnn::Tensor& sin_cache;
    const ttnn::Tensor& trans_mat;
};

using operation_attributes_t = MlaQRopeParams;
using tensor_args_t = MlaQRopeInputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::mla_q_rope::device
