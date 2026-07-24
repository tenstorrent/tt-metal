// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Forward Pass Types ==============

struct RingSDPAFwParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;  // Direction K/V is shifting in the ring
};

struct RingSDPAFwInputs {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    std::optional<ttnn::Tensor> preallocated_output;         // Preallocated output buffer
    std::optional<ttnn::Tensor> preallocated_intermediates;  // Preallocated intermediates buffer
};

using operation_attributes_t = RingSDPAFwParams;
using tensor_args_t = RingSDPAFwInputs;

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // output, intermediates

using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_sdpa_fw
