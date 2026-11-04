// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {

using ttnn::experimental::prim::MinimalMatmulConfig;

inline constexpr uint32_t kDefaultBlockTiles = 8;

struct AllGatherWeightMinimalMatmulAsyncParams {
    std::optional<MinimalMatmulConfig> config;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    uint32_t ring_size = 1;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    std::optional<uint32_t> cluster_axis;
    uint32_t num_links = 1;
    uint32_t num_workers_per_link = 3;
    uint32_t num_buffers_per_channel = 8;
    std::vector<ttnn::GlobalSemaphore> semaphores;
    bool transpose_b = true;
};

struct AllGatherWeightMinimalMatmulAsyncInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor weight_tensor;
    ttnn::Tensor persistent_weight_buffer;
};
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
