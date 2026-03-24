// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "device/k_split_gram_matmul_device_operation_types.hpp"

namespace ttml::metal {

// Compute G = X @ X^T using interleaved K-split multicast gram matmul.
// Exploits symmetry: lower triangle computes even-K, upper computes odd-K, results are accumulated.
// Uses 10x10 core grid + 10 diagonal helper cores.
//
// output_mode: UpperTriangle writes G[i,j] for i<=j. Full also writes transposed mirror G[j,i].
// math_fidelity: HiFi4 (default) or HiFi2 for faster but lower precision.
ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input,
    ops::k_split_gram_matmul::device::OutputMode output_mode =
        ops::k_split_gram_matmul::device::OutputMode::UpperTriangle,
    MathFidelity math_fidelity = MathFidelity::HiFi4);

}  // namespace ttml::metal
