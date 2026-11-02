// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Single-phase residual add fused with RMSNorm (frozen gamma): one work item per (tile-row, column slice) on one
// core each. The item's residual, delta and gamma slices are read once and stay resident in L1; the row's
// per-slice sums of squares are exchanged between the row's cores over the NoC; the sum, the normalized output
// and the rms are written from L1. Same outputs {sum, out, rms} and the same arithmetic as the two-phase
// add_rmsnorm_fw_allcore (which reads the sum back from DRAM and round-trips the partials). Requires
// rows * slices <= cores; the caller falls back to the two-phase op otherwise.
std::vector<ttnn::Tensor> add_rmsnorm_fw_rows(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& delta,
    const ttnn::Tensor& gamma,
    uint32_t num_slices,
    uint32_t slice_tiles,
    float epsilon = 1e-6F,
    bool delta_transposed = false);

}  // namespace ttml::metal
