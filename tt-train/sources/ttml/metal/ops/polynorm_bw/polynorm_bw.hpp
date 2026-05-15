// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused PolyNorm3 backward kernel entry point.
// Returns dL_dx, dL_dw ([1,1,1,3]), and dL_db ([1,1,1,1]).
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon = 1e-5F);

}  // namespace ttml::metal
