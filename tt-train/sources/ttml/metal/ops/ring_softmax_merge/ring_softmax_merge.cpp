// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_softmax_merge.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_softmax_merge(
    const ttnn::Tensor& out_acc,
    const ttnn::Tensor& lse_acc,
    const ttnn::Tensor& step_out,
    const ttnn::Tensor& step_lse,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ring_softmax_merge::RingDirection ring_direction,
    bool zigzag,
    ops::ZigzagVisitor visitor,
    AttentionMaskType mask_type) {
    return ttnn::prim::ttml_ring_softmax_merge(
        out_acc, lse_acc, step_out, step_lse, ring_size, ring_axis, step, ring_direction, zigzag, visitor, mask_type);
}

}  // namespace ttml::metal
