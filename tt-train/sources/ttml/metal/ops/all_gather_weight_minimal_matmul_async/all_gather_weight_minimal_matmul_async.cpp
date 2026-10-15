// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async.hpp"

#include "device/all_gather_weight_minimal_matmul_async_device_operation.hpp"

namespace ttml::metal {
ttnn::Tensor all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool transpose_b) {
    return ttnn::prim::ttml_all_gather_weight_minimal_matmul_async(
        input_tensor,
        weight_tensor,
        ring_size,
        ring_index,
        topology,
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        transpose_b);
}
}  // namespace ttml::metal
