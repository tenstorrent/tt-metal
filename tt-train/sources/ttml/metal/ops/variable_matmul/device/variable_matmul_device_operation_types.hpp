// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::variable_matmul::device {

struct VariableMatmulConfig {
    uint32_t M_block_size{};
    uint32_t K_block_size{};
    uint32_t N_block_size{};
    uint32_t subblock_h{};
    uint32_t subblock_w{};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};
};

struct VariableMatmulParams {
    uint32_t max_M{};  // Maximum M in elements. Determines CB sizes + core grid at compile time.
    VariableMatmulConfig config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct VariableMatmulInputs {
    ttnn::Tensor input_tensor;   // [actual_M, K] where actual_M <= max_M
    ttnn::Tensor weight_tensor;  // [K, N]
};

}  // namespace ttml::metal::ops::variable_matmul::device
