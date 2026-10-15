// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
using ttnn::experimental::prim::MinimalMatmulConfig;
struct AllGatherWeightMinimalMatmulAsyncParams {
    std::optional<MinimalMatmulConfig> config;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    uint32_t ring_size = 1;
    uint32_t ring_index = 0;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    bool transpose_b = true;
};

struct AllGatherWeightMinimalMatmulAsyncInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor weight_tensor;
};
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
