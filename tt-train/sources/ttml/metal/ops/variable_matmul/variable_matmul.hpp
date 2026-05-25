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
//   - effective_M_tiles: M tile count to actually process (0 = use input's full M).
//   - EP path: offsets_tensor + offsets_role derive M/K row+offset ranges from a
//     device tensor at runtime, letting one cached program serve all expert slices.
//
// Optional write-at-offset support (output treated as a parent buffer):
//   - output_tensor: pre-allocated output to write into. matmul-N must equal parent-N.
//     The EP path derives the per-expert row offset; without EP, the matmul writes
//     starting at row 0 of the parent.
//
// All defaults preserve "use the whole input, allocate fresh output" behavior.
// All offsets/lengths are runtime args — different values reuse the same cached program.
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
    // offsets_tensor[offsets_start_index..start_index+2] at runtime and derives the
    // matching row/K ranges (M-range for InputRow/OutputRow/InputAndOutputRow,
    // K-range for InputK/WeightK/InputAndWeightK).
    std::optional<ttnn::Tensor> offsets_tensor = std::nullopt,
    OffsetsRole offsets_role = OffsetsRole::None,
    uint32_t offsets_start_index = 0,
    // effective_M_tiles also bounds the host-side output_tensor validation on the EP
    // path (matters when token_capacity is non-tile-aligned).
    uint32_t effective_M_tiles = 0);

}  // namespace ttml::metal
