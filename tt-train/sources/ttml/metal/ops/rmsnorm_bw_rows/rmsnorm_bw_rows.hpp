// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Single-phase RMSNorm backward with a frozen gamma: one work item per (tile-row, column slice) on one core
// each; the item's slices of input, gamma and dL_dout stay resident in L1 for both passes (the row dot product
// and the gradient), the row's per-slice dot products are exchanged between the row's cores over the NoC. Reads
// every operand once (the two-phase rmsnorm_bw_allcore reads input and dL_dout twice and round-trips partials
// through DRAM). Requires rows * slices <= cores; the caller falls back to the two-phase op otherwise.
// With transposed_grad the result also holds dL_dinput^T [1, 1, C, S] (iteration 37: the gradient of a transposed
// delta), written tile by tile as the row's blocks are produced.
std::vector<ttnn::Tensor> rmsnorm_bw_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& rms,
    const ttnn::Tensor& dL_dout,
    uint32_t num_slices,
    uint32_t slice_tiles,
    const std::optional<ttnn::Tensor>& residual_grad = std::nullopt,
    bool transposed_grad = false);

}  // namespace ttml::metal
