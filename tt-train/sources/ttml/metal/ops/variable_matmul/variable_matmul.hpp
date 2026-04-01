// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "device/variable_matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttml::metal {

// Re-export config type for convenience
using VariableMatmulConfig = ttml::metal::ops::variable_matmul::device::VariableMatmulConfig;

// Variable-M matmul: compiles at most 2 programs (one per transpose variant),
// then dispatches any M shape without recompilation.
ttnn::Tensor variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttml::metal
