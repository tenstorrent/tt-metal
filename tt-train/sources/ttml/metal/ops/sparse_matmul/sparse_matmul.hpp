// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {

// Grouped (per-expert) matmul for tokens packed by the MoE group op.
//
// Shapes:
//   X       : [1, 1, T_cap, K]   bf16 TILE DRAM
//   W       : [E_local, K, N]    bf16 TILE DRAM
//   offsets : [E_local + 1]      uint32 L1 (or DRAM)
// Returns:
//   Y       : [1, 1, T_cap, N]   bf16 TILE DRAM
//
// Two implementations selected at dispatch time based on DRAM bank alignment:
//
//   * Fast path (zero-copy views via ttnn::narrow): pre-allocates a zeroed
//     output and per-expert narrows X, W and the output into place so
//     ttnn_fixed::matmul writes directly into the final buffer. No slice
//     copies, no concat. Requires:
//        (TILE_H * H) % (num_banks * TILE_HW) == 0
//        (TILE_H * N) % (num_banks * TILE_HW) == 0
//        (K * N)      % (num_banks * TILE_HW) == 0
//
//   * Slow path (ttnn::slice per expert + ttnn::concat): always valid.
//     Used when the alignment conditions above do not hold — notably on
//     P100 (7 DRAM banks) for power-of-2 H that do not contain a factor
//     of 7.
//
// The decision is static: for tile-aligned `offsets` (guaranteed by the
// group op) the worst-case row offset is 32, so checking the 32-multiple
// form above is both necessary and sufficient for every non-empty expert.
ttnn::Tensor sparse_matmul(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets);

// Debug counters — number of sparse_matmul calls that took each path.
// Used by tests to assert the expected path ran on the current hardware.
struct SparseMatmulCounters {
    uint64_t fast_path_calls;
    uint64_t slow_path_calls;
};
SparseMatmulCounters get_sparse_matmul_counters();
void reset_sparse_matmul_counters();

}  // namespace ttml::metal
