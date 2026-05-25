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

// Variable-M and variable-K matmul: M and K are runtime args. The program cache
// keys on (N, transpose flags, grid), so a single cached program services any
// (M, K) pair within a transpose variant.
//
// Optional read-at-offset support (input/weight treated as parent buffers):
//   - in0_row_offset_tiles: tile offset on the in0 matmul-M axis.
//   - effective_M_tiles: M tile count to actually process (0 = use input's full M).
//   - in0_k_offset_tiles: tile offset on the in0 matmul-K axis. The K count comes
//     from the weight (no effective_K argument); in0 is read as if it had a larger
//     K extent and we slice [k_offset, k_offset + K) tiles.
//   - in1_k_offset_tiles: tile offset on the in1 matmul-K axis. When set, in0's K
//     determines matmul-K; the weight is read as if it had a larger K extent and
//     we slice [k_offset, k_offset + K) tiles. Cannot be combined with in0_k_offset.
//
// Optional write-at-offset support (output treated as a parent buffer):
//   - output_tensor: pre-allocated output to write into.
//   - out_row_offset_tiles: tile offset on the output's M axis. matmul-N must equal
//     parent-N. When omitted, a fresh output tensor is allocated and returned.
//
// All defaults preserve "use the whole input, allocate fresh output" behavior.
// All offsets/lengths are runtime args — different values reuse the same cached program.
//
// Tile alignment: all offsets/counts must be in TILE_HEIGHT (32) units. With transpose_a,
// "row" still means the matmul-M axis (= input's stored *col* axis) and "k_offset" still
// means the matmul-K axis (= input's stored *row* axis).
using OffsetsRole = ttml::metal::ops::variable_matmul::device::OffsetsRole;

ttnn::Tensor variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // Write-at-offset output (optional). When set, matmul writes into a row range of
    // this caller-provided parent. EP path overrides the row offset via offsets_tensor.
    std::optional<ttnn::Tensor> output_tensor = std::nullopt,
    // EP path: when offsets_tensor is set, the dataflow kernel reads
    // offsets_tensor[offsets_start_index..start_index+2] at runtime and overrides the
    // matching scalar offsets below (M-range for InputRow/OutputRow/InputAndOutputRow,
    // K-range for InputK/WeightK/InputAndWeightK).
    std::optional<ttnn::Tensor> offsets_tensor = std::nullopt,
    OffsetsRole offsets_role = OffsetsRole::None,
    uint32_t offsets_start_index = 0,
    // Scalar (host-known) offsets — used only when offsets_tensor is std::nullopt or
    // the role does not cover the relevant axis. EP path overrides these.
    uint32_t in0_row_offset_tiles = 0,
    uint32_t effective_M_tiles = 0,
    uint32_t in0_k_offset_tiles = 0,
    uint32_t in1_k_offset_tiles = 0,
    uint32_t out_row_offset_tiles = 0);

}  // namespace ttml::metal
