// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Causal scaled-dot-product-attention backward on a cyclic schedule, with the
// row packet relayed between cores rather than re-read from DRAM, and without
// atomic read-modify-write on dQ.
//
// The schedule ties cores to sequence length: a block is
// rows_per_block_tiles * 32 rows, there are T = 2C of them, so one schedule
// covers N = 2 * C * rows_per_block_tiles * 32 rows and C follows from N.
// Every (batch, head) slice is an independent problem, and they run
// side by side on disjoint rectangles of the grid -- which is what fills a
// grid when the sequence is shorter than one schedule needs.
//
// Requires causal masking, a head dimension that is a multiple of 32, and a
// sequence length that the schedule divides. The statistics are taken as
// given rather than recomputed: log_sum_exp and row_scalar come from the
// forward pass.
//
// Returns dQ, dK, dV in Float32.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t rows_per_block_tiles = 1U,
    bool use_barrier = false);

}  // namespace ttml::metal
