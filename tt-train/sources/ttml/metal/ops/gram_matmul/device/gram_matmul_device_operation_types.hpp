// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gram_matmul::device {

struct GramMatmulConfig {
    uint32_t M_block_size{};
    uint32_t K_block_size{};
    uint32_t N_block_size{};
    uint32_t subblock_h{};
    uint32_t subblock_w{};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};
};

struct operation_attributes_t {
    std::optional<GramMatmulConfig> config;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    ttnn::Tensor input_tensor;
    std::optional<ttnn::Tensor> output_tensor;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::gram_matmul::device
