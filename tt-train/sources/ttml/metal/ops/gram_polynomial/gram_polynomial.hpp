// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "metal/common/const_utils.hpp"

namespace ttml::metal {

// Compute bG + cG² for symmetric G using interleaved K-split multicast.
// G must be square [M, M] with even M (in tiles). Output is symmetric [M, M].
// Exploits symmetry: lower triangle computes even-K partial, upper computes odd-K, then reduce.
//
// output_mode: UpperTriangle writes result[i,j] for i<=j. Full also writes transposed mirror.
ttnn::Tensor gram_polynomial(
    const ttnn::Tensor& G,
    float b,
    float c,
    OutputMode output_mode = OutputMode::UpperTriangle,
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttml::metal
