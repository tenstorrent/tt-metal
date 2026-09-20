// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_softmax_merge_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Merge one ring step's partial attention (step_out bf16, step_lse Float32
// with the row's log-sum-exp in column 0) into the running Float32
// accumulators out_acc and lse_acc, in place, by the online-softmax rule, on
// the chips the step selects (the zigzag visitor rule, or the contiguous
// causal rule with zigzag = false). Returns the two accumulators.
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_softmax_merge(
    const ttnn::Tensor& out_acc,
    const ttnn::Tensor& lse_acc,
    const ttnn::Tensor& step_out,
    const ttnn::Tensor& step_lse,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ring_softmax_merge::RingDirection ring_direction = ops::ring_softmax_merge::RingDirection::Backward,
    bool zigzag = true,
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any,
    AttentionMaskType mask_type = AttentionMaskType::Causal);

}  // namespace ttml::metal
