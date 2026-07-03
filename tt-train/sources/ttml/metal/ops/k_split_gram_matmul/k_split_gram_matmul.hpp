// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "metal/common/const_utils.hpp"

namespace ttml::metal {

// Compute G = X @ X^T using a K-split multicast gram matmul.
//
// Gram matrix multiplication is symmetric (G[i,j] = G[j,i]^T), so a standard matmul wastes half
// the compute. This op exploits the symmetry by computing each output block exactly once.
//
// Constraints: input X is [M, K] in tile layout, BF16, DRAM. K must be a multiple of 64
// (K_tiles must be even — enforced by TT_FATAL in the device op validation).
//
// Core grid (N = min(device_grid.x - 1, device_grid.y)): an N × N compute grid plus a column
// of diagonal helpers at x = N.
//
//             x=0     x=1     x=2   ...   x=N-1     x=N (diagonal helpers)
//   y=0     [ 0,0 ] [ 1,0 ] [ 2,0 ]      [N-1,0]    [ H0 ]
//   y=1     [ 0,1 ] [ 1,1 ] [ 2,1 ]      [N-1,1]    [ H1 ]
//   y=2     [ 0,2 ] [ 1,2 ] [ 2,2 ]      [N-1,2]    [ H2 ]
//    :
//   y=N-1   [0,N-1] [1,N-1] [2,N-1]    [N-1,N-1]    [HN-1]
//
// Each tile of the output block at (x, y) corresponds to a slice of G. We use Mpc = ceil(M_tiles / N)
// to denote the number of rows of X assigned to each grid row (and columns to each grid column);
// padding makes Mpc uniform. M_block ≤ Mpc is the streaming granularity inside a core.
//
// Core roles (cores at x=0 and y=0 inject data AND compute):
//   * Column x=0:   Row injectors  — read Mpc rows of X from DRAM, multicast to their row.
//   * Row y=0:      Column injectors — read Mpc rows of X from DRAM, multicast down their column.
//   * x < y:        Lower triangle — computes even-K partial.
//   * x > y:        Upper triangle — computes odd-K partial.
//   * x = y:        Diagonal — computes even-K partial.
//   * x = N:        Diagonal helpers — compute odd-K partial (read odd K-columns from DRAM
//                   independently; keeps the diagonal cores from doing 2× work).
//
// Data flow:
//   1. Input streaming: row injector at (0, y) reads M_block rows of X and multicasts to all
//      cores in row y (including the helper at (N, y)). Column injector at (x, 0) reads M_block
//      rows and multicasts down column x. Each core thus receives two streams: a row block (in0)
//      and a column block (in1).
//   2. K-column split: even K-columns go to lower triangle + diagonal; odd K-columns go to upper
//      triangle + diagonal helpers. Blocks alternate: even block → lower multicast, odd block →
//      upper multicast.
//   3. Compute: each core runs matmul on its received tiles (in0 · in1^T), accumulating across
//      K-blocks. The output per core is an M_block × N_block partial of the Mpc × Mpc output
//      block at the core's grid position.
//   4. Reduction: lower core (x, y) sends its partial to upper core (y, x) via NOC write. The
//      upper core adds: G[j, i] = G_odd[j, i] + G_even[i, j]^T (transpose+add is fused in the
//      compute kernel). Diagonal cores send to their helper, which performs the same add. Only
//      upper triangle + helpers hold the final result.
//   5. Output write: upper triangle cores and helpers write their blocks to DRAM. In Full mode,
//      they also write the transposed mirror tile to the lower triangle position.
//
// output_mode: UpperTriangle only writes G[i,j] for i<=j. Full also writes transposed mirror G[j,i].
ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input,
    OutputMode output_mode = OutputMode::UpperTriangle,
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttml::metal
