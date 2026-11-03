// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "device/all_gather_weight_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {
ttnn::Tensor all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_weight_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t num_links = 1,
    uint32_t num_workers_per_link = 3,
    uint32_t num_buffers_per_channel = 8,
    bool transpose_b = true);
}
