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
// input @ weight, where the weight is sharded along N (one stripe per device on cluster_axis) and gathered
// over fabric while the matmul runs.
//
// persistent_weight_buffer is full-weight-sized workspace for the remote stripes; the local stripe is read straight
// from weight_tensor and never copied into it. Neighbours write into it from the start of a call and can be one call
// ahead, so: allocate two buffers once and keep them, alternate them between back-to-back calls on the same ring,
// and don't fill, free or reuse them in between.
//
// multi_device_global_semaphore holds one semaphore per ring device, all distinct, starting at 0 and used by this op
// only. They are never reset: each call takes back only its own counts. One set serves calls of any shape, but not
// calls that differ in num_links, topology, cluster_axis or grid, unless the device is synchronized in between.
ttnn::Tensor all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_weight_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t cluster_axis,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t num_links = 1,
    bool transpose_b = true);
}
