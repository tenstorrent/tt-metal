// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_bw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

struct RingCyclicSDPABackwardParams {
    uint32_t ring_size{};
    uint32_t ring_axis{};
    uint32_t step{};
    // The mask of the *whole* attention problem, not of this step's chunk
    // pair. Causal means the ring skips the steps whose visiting chunk is
    // later than the local one, runs the diagonal step with the triangular
    // schedule, and every earlier step with the dense one.
    AttentionMaskType mask_type{AttentionMaskType::Causal};
    RingDirection ring_direction{RingDirection::Backward};

    uint32_t rows_per_block_tiles{1U};
    bool use_barrier{false};
    // Add this step's contribution into the preallocated outputs, which the
    // caller passes as its running accumulators, instead of writing a fresh
    // per-step gradient the caller must then add. A chip the causal schedule
    // skips has no program and leaves them untouched, which is exactly right.
    bool accumulate_into_outputs{false};

    // Contiguous or Zigzag; see ops::RingLayout. Under Zigzag every local
    // tensor is two chunks back to back, mask_type names the launch -- Causal
    // for the two triangles of the diagonal step, None for the full blocks of
    // any step -- and each chip runs the chunk pairs zigzag_sub_problems
    // gives it as slices of one program.
    ops::RingLayout layout{ops::RingLayout::Contiguous};
};

struct RingCyclicSDPABackwardInputs {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor grad_output;
    // Global statistics from the forward, one value per row in column 0 of a
    // tile: the log-sum-exp and D = rowsum(dO . O). Global, not per chunk --
    // the ring's partial contributions only sum to the right gradients if
    // every step recomputes P against the global normaliser.
    ttnn::Tensor log_sum_exp;
    ttnn::Tensor row_scalar;

    std::optional<ttnn::Tensor> preallocated_grad_query;
    std::optional<ttnn::Tensor> preallocated_grad_key;
    std::optional<ttnn::Tensor> preallocated_grad_value;
};

using operation_attributes_t = RingCyclicSDPABackwardParams;
using tensor_args_t = RingCyclicSDPABackwardInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_bw
