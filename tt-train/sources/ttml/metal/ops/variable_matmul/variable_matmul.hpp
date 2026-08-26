// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// Variable-M and variable-K matmul: M and K are runtime args, excluded from the program hash
// (see compute_program_hash for the full key), so one cached program serves any (M, K) pair.
//
// Always reads offsets on-device: every call provides an offsets_tensor; the offset role is
// fixed by which entry point is called (see below). The dataflow kernels read
// offsets_tensor[offsets_start_index..start_index+2] at runtime and derive the per-call M/K
// ranges (and, for variable_matmul_into_rows, the output write-at-offset row). One cached
// program serves all expert slices.
//
// CONTRACT: offsets values MUST be tile-aligned (multiples of TILE_HEIGHT = 32). The kernels
// address the read/write/K windows at tile granularity (offset / 32), so a non-multiple-of-32
// offset cannot be represented — it would silently start on the wrong tile and overlap a
// neighbouring slice. Callers (e.g. moe_ffn) must pad each expert's row range to a tile
// boundary. Debug builds assert this in the dataflow kernels.
//
// On-device offsets are what makes the op viable under EP-sharded MoE — when the per-expert
// dispatch counts live on the device, a host scalar would require an all-gather of offsets
// across the EP dim. This is the motivation; the contract works the same on a single device.
//
// Two entry points, one per offset interpretation (the output contract is encoded in the
// signature rather than a role enum + optional, so illegal combinations can't be expressed):
//
//   variable_matmul_into_rows — read in0 row-range [offsets[start], offsets[start+1]) and write
//     the result into the SAME rows of the caller-provided output_tensor (a shared parent buffer
//     that successive calls fill in place). matmul-N must equal output_tensor's N.
//   variable_matmul_k_sliced — K-slice BOTH operands to [offsets[start], offsets[start+1]) and
//     reduce only over that range; returns a freshly allocated [M, N].

// Write-at-offset matmul; result lands in output_tensor rows [offsets[start], offsets[start+1]).
// Returns output_tensor.
ttnn::Tensor variable_matmul_into_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    const ttnn::Tensor& output_tensor,
    uint32_t offsets_start_index = 0,
    // Per-call matmul-M extent in tiles: a build-time hint for the grid orientation only (the
    // actual rows processed come from offsets). See the expected_M_tiles doc in the types header.
    uint32_t expected_M_tiles = 0,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

// K-sliced matmul; reduces over the offsets[start..start+2] K-range of both operands and returns
// a freshly allocated [M, N].
ttnn::Tensor variable_matmul_k_sliced(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    uint32_t offsets_start_index = 0,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttml::metal
