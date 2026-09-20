// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_fw::device {

// The forward pass on the cyclic schedule (tt-flash-attn, Algorithm 8). The
// schedule, the grid layout and the packet transport are those of
// cyclic_sdpa_bw; what travels in the packet is the query block and the
// online-softmax state (O^T, m, l) instead of (dO, L, D, dQ^T), and the
// columns carry no outputs.
struct CyclicSDPAForwardParams {
    // Rows per block, in tiles; C = N / (2 * rows_per_block_tiles * 32).
    uint32_t rows_per_block_tiles{1U};

    // Causal is the paper's triangle; None the unmasked two-pass schedule a
    // ring step needs against an earlier chunk. Arbitrary is rejected.
    ttml::metal::AttentionMaskType mask_type{ttml::metal::AttentionMaskType::Causal};

    // Cap on the number of groups running side by side (0 = as many as fit).
    uint32_t max_groups{0U};

    // Sub-problems as chunk pairs of the local sequence; see the backward's
    // attributes. Sub-problem p attends the query rows of chunk row_chunks[p]
    // to the keys of chunk col_chunks[p]; each is one more slice. Two pairs
    // of one launch may not share a row chunk (they would both write its
    // output) -- sharing a column chunk is fine here, since columns have no
    // outputs.
    uint32_t sequence_chunks{1U};
    std::vector<uint32_t> row_chunks{};
    std::vector<uint32_t> col_chunks{};
};

struct CyclicSDPAForwardInputs {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;

    // The attention output, the query's shape, bfloat16.
    std::optional<ttnn::Tensor> preallocated_output;
    // The per-row log-sum-exp of the scaled, masked scores as the tt-train
    // convention has it: (batch, heads, sequence, 32) Float32, column 0.
    std::optional<ttnn::Tensor> preallocated_intermediates;
};

using operation_attributes_t = CyclicSDPAForwardParams;
using tensor_args_t = CyclicSDPAForwardInputs;
// output, intermediates, and two Float32 scratch tensors the kernels spill
// the unfinished state into between a row's streaks: the O^T accumulator
// (the query's shape) and the statistics m, l as two tiles per row tile
// (batch, heads, sequence, 64). The caller has no use for the last two.
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::cyclic_sdpa_fw::device
