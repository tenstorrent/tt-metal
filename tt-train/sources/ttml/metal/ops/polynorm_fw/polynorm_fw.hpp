// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused PolyNorm forward kernel entry point.
ttnn::Tensor polynorm_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& weight, const ttnn::Tensor& bias, float epsilon = 1e-5F);

}  // namespace ttml::metal
