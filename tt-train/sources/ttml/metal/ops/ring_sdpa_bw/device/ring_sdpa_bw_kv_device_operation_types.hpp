// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Backward KV Types ==============

struct RingSDPABwKVParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;   // Direction K/V is shifting in the ring
    RingLaneSelection lane_selection = RingLaneSelection::None;  // See RingLaneSelection in ring_sdpa_utils.hpp
};

// Second-lane tensors for RingLaneSelection::ZigZag. Devices assigned lane 1 at a step read
// these and write grad_key / grad_value in place; u_scaler is the shared tensor produced
// by the Q op for the lane each device computes.
struct RingSDPABwKVLane {
    ttnn::Tensor grad_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_key;
    ttnn::Tensor grad_value;
};

struct RingSDPABwKVInputs {
    ttnn::Tensor grad_output;
    ttnn::Tensor u_scaler;  // Precomputed rowsum(dO * O) from Q kernel
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    std::optional<ttnn::Tensor> preallocated_grad_key;    // Preallocated output buffer
    std::optional<ttnn::Tensor> preallocated_grad_value;  // Preallocated output buffer
    std::optional<RingSDPABwKVLane> lane1;                // Required iff lane_selection == ZigZag
};

using operation_attributes_t = RingSDPABwKVParams;
using tensor_args_t = RingSDPABwKVInputs;

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // [grad_K, grad_V]

using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv
