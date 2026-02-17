// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "device/gram_polynomial_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Compute H = b*G + c*G^2 where G = X @ X^T.
// X must be a BFLOAT16 tensor with cols >= rows in the last two dimensions.
// Phase 1: G = gram_matmul(X) (existing op, unchanged).
// Phase 2: H = c*G*G + b*G (new device op on the full grid, no transpose).
// Output is a square tensor [..., M, M] where M = X.shape[-2].
ttnn::Tensor gram_polynomial(
    const ttnn::Tensor& input_tensor,
    float b,
    float c,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const tt::tt_metal::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttml::metal
