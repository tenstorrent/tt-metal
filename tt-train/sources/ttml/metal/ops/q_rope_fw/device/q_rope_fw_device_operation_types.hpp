// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::q_rope_fw::device {

struct QRopeFwParams {
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
};

struct QRopeFwInputs {
    const ttnn::Tensor& q_in;
    const ttnn::Tensor& cos_cache;
    const ttnn::Tensor& sin_cache;
    const ttnn::Tensor& trans_mat;
};

using operation_attributes_t = QRopeFwParams;
using tensor_args_t = QRopeFwInputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::q_rope_fw::device
