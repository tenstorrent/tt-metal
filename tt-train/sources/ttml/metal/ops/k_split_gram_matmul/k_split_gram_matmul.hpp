// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "metal/common/const_utils.hpp"

namespace ttml::metal {

// Compute G = X @ X^T using interleaved K-split multicast gram matmul.
// Exploits symmetry: lower triangle computes even-K, upper computes odd-K, results are accumulated.
// Uses min(device_grid.x - 1, device_grid.y) core grid + column of diagonal helper cores.
//
// output_mode: UpperTriangle writes G[i,j] for i<=j. Full also writes transposed mirror G[j,i].
ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input,
    OutputMode output_mode = OutputMode::UpperTriangle,
    MathFidelity math_fidelity = MathFidelity::HiFi4);

}  // namespace ttml::metal
