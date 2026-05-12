// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "device/variable_matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttml::metal {

// Re-export config type for convenience
using VariableMatmulConfig = ttml::metal::ops::variable_matmul::device::VariableMatmulConfig;

// Variable-M matmul: compiles at most 2 programs (one per transpose variant),
// then dispatches any M shape without recompilation.
//
// Optional read-at-offset support: `in0_row_offset_tiles` is added to in0 tile addresses
// (treats input_tensor as a parent buffer), and `effective_M_tiles` overrides the M tile
// count that's actually processed (0 = use input's full M). Together they let the caller
// process a sub-range of the parent tensor without materializing a slice. These are
// runtime args — different (offset, length) values reuse the same cached program.
//
// Tile alignment: both must be in TILE_HEIGHT (32) units. With transpose_a, "row" still
// means the matmul-M axis (which maps to the input's stored *col* axis).
ttnn::Tensor variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t in0_row_offset_tiles = 0,
    uint32_t effective_M_tiles = 0);

}  // namespace ttml::metal
