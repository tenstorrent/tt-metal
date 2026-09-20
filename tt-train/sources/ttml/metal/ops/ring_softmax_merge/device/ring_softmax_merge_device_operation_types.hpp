// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_softmax_merge {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// The online-softmax merge of one ring step's partial attention into the
// running accumulators, in place and in one launch, on the chips the step
// selects. Per query row with running (O, lse) and the step's (O_s, lse_s):
//
//     m = max(lse, lse_s);  w = exp(lse - m) / (exp(lse - m) + exp(lse_s - m))
//     O   <- w O + (1 - w) O_s
//     lse <- m + ln(exp(lse - m) + exp(lse_s - m))
//
// which is what the ring driver did in some fourteen elementwise ttnn ops a
// partial. A running lse of -inf (no contribution yet) gives w = 0 and takes
// the step's values. Chips the step does not select run nothing, so no
// "no contribution" fill of the step's lse is needed either.
struct RingSoftmaxMergeParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    RingDirection ring_direction = ttnn_fixed::distributed::RingShiftDirection::Backward;
    // Zigzag: the chips `visitor` selects. Contiguous: the causal rule's chips.
    bool zigzag = true;
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
};

struct RingSoftmaxMergeInputs {
    ttnn::Tensor out_acc;   // (B, H, S, d) Float32, updated in place
    ttnn::Tensor lse_acc;   // (B, H, S, 32) Float32, column 0, updated in place
    ttnn::Tensor step_out;  // (B, H, S, d) bfloat16, the step's normalised output
    ttnn::Tensor step_lse;  // (B, H, S, 32) Float32, column 0, the step's log-sum-exp
};

using operation_attributes_t = RingSoftmaxMergeParams;
using tensor_args_t = RingSoftmaxMergeInputs;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // out_acc, lse_acc
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_softmax_merge
