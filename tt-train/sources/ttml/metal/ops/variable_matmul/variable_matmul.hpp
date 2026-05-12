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
// Optional read-at-offset support:
//   - in0_row_offset_tiles: tile offset on the in0 matmul-M axis (input treated as a
//     parent buffer).
//   - effective_M_tiles: M tile count to actually process (0 = use input's full M).
//   - in0_k_offset_tiles: tile offset on the in0 matmul-K axis. The K count comes from
//     the weight (no effective_K argument); in0 is read as if it had a larger K extent
//     and we slice [k_offset, k_offset + K) tiles.
// All defaults preserve "use the whole input" behavior. These are runtime args —
// different offset/length values reuse the same cached program.
//
// Tile alignment: all offsets/counts must be in TILE_HEIGHT (32) units. With transpose_a,
// "row" still means the matmul-M axis (= input's stored *col* axis) and "k_offset" still
// means the matmul-K axis (= input's stored *row* axis).
ttnn::Tensor variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t in0_row_offset_tiles = 0,
    uint32_t effective_M_tiles = 0,
    uint32_t in0_k_offset_tiles = 0);

}  // namespace ttml::metal
