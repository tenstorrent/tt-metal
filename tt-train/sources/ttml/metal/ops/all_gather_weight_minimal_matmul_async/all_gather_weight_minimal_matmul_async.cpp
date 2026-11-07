// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async.hpp"

#include "device/all_gather_weight_minimal_matmul_async_device_operation.hpp"

namespace ttml::metal {
ttnn::Tensor all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_weight_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t cluster_axis,
    ttnn::ccl::Topology topology,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_links,
    bool transpose_b) {
    return ttnn::prim::ttml_all_gather_weight_minimal_matmul_async(
        input_tensor,
        weight_tensor,
        persistent_weight_buffer,
        multi_device_global_semaphore,
        cluster_axis,
        topology,
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        num_links,
        transpose_b);
}
}  // namespace ttml::metal
