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

// Variable-M and variable-K matmul: M and K are runtime args. The program cache keys
// on (N, transpose flags, grid), so a single cached program services any (M, K) pair
// within a transpose variant.
//
// EP-only op: every call must provide an offsets_tensor and an OffsetsRole. The dataflow
// kernels read offsets_tensor[offsets_start_index..start_index+2] at runtime and derive
// the per-call M/K ranges (and, for InputAndOutputRow, the output write-at-offset row).
// One cached program serves all expert slices.
//
// Roles:
//   - InputAndOutputRow: read in0 row-range [offsets[start], offsets[start+1]) and write
//     output rows in the same range. Requires a caller-provided output_tensor (the parent
//     buffer that the matmul writes into). matmul-N must equal output_tensor's N.
//   - InputAndWeightK: K-slice both in0 and in1 to [offsets[start], offsets[start+1]).
//     Output is freshly allocated.
using OffsetsRole = ttml::metal::ops::variable_matmul::device::OffsetsRole;

ttnn::Tensor variable_matmul(
    // Required.
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    OffsetsRole offsets_role,
    // Optional from here on.
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // Required when offsets_role == InputAndOutputRow (caller-provided parent buffer that
    // the matmul writes into); ignored for InputAndWeightK (output is freshly allocated).
    std::optional<ttnn::Tensor> output_tensor = std::nullopt,
    uint32_t offsets_start_index = 0,
    // Bounds the host-side output_tensor validation on the EP path (matters when
    // token_capacity is non-tile-aligned). Also hints the transpose_core_grid heuristic.
    uint32_t effective_M_tiles = 0);

}  // namespace ttml::metal
