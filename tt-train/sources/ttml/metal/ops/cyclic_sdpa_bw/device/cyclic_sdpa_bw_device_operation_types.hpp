// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

struct CyclicSDPABackwardParams {
    // Rows per block, in tiles: a block is rows_per_block_tiles * 32 rows
    // tall. It sets how much arithmetic one block pair carries, and the
    // number of cores one schedule needs follows from it and the sequence
    // length: C = N / (2 * rows_per_block_tiles * 32).
    uint32_t rows_per_block_tiles{1U};

    // Algorithm 3 rather than Algorithm 4: order the timesteps with a
    // chip-wide barrier instead of the endpoint counters. Both compute the
    // same gradients, bit for bit; the barrier variant exists because when
    // the counters deadlock, flipping this says in one run whether the
    // endpoint protocol or something underneath it is at fault.
    bool use_barrier{false};
};

struct CyclicSDPABackwardInputs {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const ttnn::Tensor& grad_output;
    // The forward's per-row statistics, one value per row in column 0 of a
    // tile: the log-sum-exp, and D = rowsum(dO . O).
    const ttnn::Tensor& log_sum_exp;
    const ttnn::Tensor& row_scalar;

    std::optional<ttnn::Tensor> preallocated_grad_query;
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;
};

using operation_attributes_t = CyclicSDPABackwardParams;
using tensor_args_t = CyclicSDPABackwardInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::cyclic_sdpa_bw::device
