// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ring_zigzag_sdpa: the ring_sdpa_fw / ring_sdpa_bw mesh wrapper over the
// single-chip SDPA kernels, with the chips that run chosen by where this
// step's visiting chunk comes from (ops::ZigzagVisitor) instead of by the
// contiguous layout's causal rule. The kernels and the reference ring ops
// are untouched; this exists so the zigzag layout can use them on one chunk
// pair per launch.

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_zigzag_bw::q {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Backward Q Types ==============

struct RingZigzagBwQParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;
    // Which chips run this launch; see ops::ZigzagVisitor.
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any;  // Direction K/V is shifting in the ring
};

struct RingZigzagBwQInputs {
    ttnn::Tensor grad_output;
    ttnn::Tensor attn_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    std::optional<ttnn::Tensor> preallocated_grad_query;  // Preallocated output buffer
};

using operation_attributes_t = RingZigzagBwQParams;
using tensor_args_t = RingZigzagBwQInputs;

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // [grad_Q, u_scaler]

using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_zigzag_bw::q
