// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "device/minimal_matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {
ttnn::Tensor minimal_matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    bool bidirectional = true);
}
