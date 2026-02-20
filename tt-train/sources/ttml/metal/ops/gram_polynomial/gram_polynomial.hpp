// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "device/gram_polynomial_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Newton-Schulz iteration(s): X' = aX + (cG^2 + bG)X where G = XX^T
// When use_trace=true, captures a trace and replays it, eliminating
// per-iteration host overhead. Requires trace_region_size > 0 when opening the device.
ttnn::Tensor newton_schulz(
    const ttnn::Tensor& x_tensor,
    float a,
    float b,
    float c,
    int num_iterations = 1,
    bool use_trace = false,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const tt::tt_metal::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttml::metal
