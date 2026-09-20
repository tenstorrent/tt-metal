// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_ttnn_sdpa_fw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// One step of a ring-attention forward through ttnn's chunk-blocked
// flash-attention kernel (ttnn::transformer::scaled_dot_product_attention
// with its lse output), on the chips the step selects: the contiguous
// layout's causal rule (get_device_execution_info) or the zigzag layout's
// visitor rule (zigzag_visitor_runs). Drop-in for ring_sdpa_fw and
// ring_zigzag_sdpa_fw: the output in the query's dtype, the intermediates a
// (B, H, S, 32) Float32 tensor with lse = scale * max + ln sum in column 0,
// and chips that run nothing leave both preallocated tensors as they are.
struct RingTtnnSdpaFwParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
    RingDirection ring_direction = ttnn_fixed::distributed::RingShiftDirection::Backward;
    // Zigzag: the launch runs on the chips `visitor` selects with `mask_type`
    // as given. Contiguous: the causal rule decides per chip whether to run
    // and with which mask (causal on the diagonal step, none before it).
    bool zigzag = false;
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any;
    // ttnn's query and key chunk, in rows; clamped to the local sequence.
    uint32_t chunk_size = 256;
};

struct RingTtnnSdpaFwInputs {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    std::optional<ttnn::Tensor> preallocated_output;
    std::optional<ttnn::Tensor> preallocated_intermediates;
};

using operation_attributes_t = RingTtnnSdpaFwParams;
using tensor_args_t = RingTtnnSdpaFwInputs;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // output, intermediates
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_ttnn_sdpa_fw
