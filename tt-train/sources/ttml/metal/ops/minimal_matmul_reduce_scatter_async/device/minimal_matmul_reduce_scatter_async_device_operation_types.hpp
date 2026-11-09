// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
using ttnn::experimental::prim::MinimalMatmulConfig;

inline constexpr uint32_t kDefaultBlockTiles = 8;

struct MinimalMatmulReduceScatterAsyncParams {
    std::optional<MinimalMatmulConfig> config;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    uint32_t ring_size = 1;
    std::optional<uint32_t> cluster_axis;
    bool transpose_a = false;
    bool transpose_b = false;
};

struct MinimalMatmulReduceScatterAsyncInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor weight_tensor;
    ttnn::Tensor persistent_buffer;
};
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async
