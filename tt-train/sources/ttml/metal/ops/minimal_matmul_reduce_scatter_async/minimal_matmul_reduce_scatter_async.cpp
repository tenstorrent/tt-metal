// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_reduce_scatter_async.hpp"

#include "device/minimal_matmul_reduce_scatter_async_device_operation.hpp"

namespace ttml::metal {
ttnn::Tensor minimal_matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool transpose_a,
    bool transpose_b,
    bool bidirectional) {
    return ttnn::prim::ttml_minimal_matmul_reduce_scatter_async(
        input_tensor,
        weight_tensor,
        persistent_buffer,
        multi_device_global_semaphore,
        cluster_axis,
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        transpose_a,
        transpose_b,
        bidirectional);
}
}  // namespace ttml::metal
