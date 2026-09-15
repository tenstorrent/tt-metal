// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

struct CyclicSDPABackwardParams {
    // Rows per block, in tiles: a block is rows_per_block_tiles * 32 rows
    // tall. It sets how much arithmetic one block pair carries, and the
    // number of cores one schedule needs follows from it and the sequence
    // length: C = N / (2 * rows_per_block_tiles * 32).
    uint32_t rows_per_block_tiles{1U};

    // Which block pairs the schedule covers. Causal is the paper's triangle.
    // None is the unmasked schedule: every pair is live, which is what a
    // ring-attention step needs when the visiting key/value chunk is earlier
    // in the sequence than the local query chunk. Arbitrary is rejected --
    // there is no mask tensor path here.
    ttml::metal::AttentionMaskType mask_type{ttml::metal::AttentionMaskType::Causal};

    // Algorithm 3 rather than Algorithm 4: order the timesteps with a
    // chip-wide barrier instead of the endpoint counters. Both compute the
    // same gradients, bit for bit; the barrier variant exists because when
    // the counters deadlock, flipping this says in one run whether the
    // endpoint protocol or something underneath it is at fault.
    bool use_barrier{false};

    // Accumulate into the outputs rather than overwrite them: every gradient
    // starts from what the output buffer holds. The caller must then pass
    // preallocated outputs carrying its running sums. dQ always behaves this
    // way; this extends it to dK and dV, whose first visit otherwise starts
    // from zero without reading it.
    bool accumulate_into_outputs{false};

    // Cap on the number of groups running side by side (0 = as many as fit).
    // The groups run the remaining slices in turn either way; the cap exists
    // so a test can force that loop at a size where every slice would fit.
    uint32_t max_groups{0U};
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
