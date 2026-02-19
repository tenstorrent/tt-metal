// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "device/gram_matmul_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Compute the Gram matrix G = X @ X^T.
// X must be a BFLOAT16 tensor with cols >= rows in the last two dimensions.
// Output is a square tensor [... , M, M] where M = X.shape[-2].
ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input_tensor,
    const std::optional<const ttml::metal::ops::gram_matmul::device::GramMatmulConfig>& config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const tt::tt_metal::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

}  // namespace ttml::metal
