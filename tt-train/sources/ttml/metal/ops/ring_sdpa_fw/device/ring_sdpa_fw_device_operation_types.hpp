// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

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

    static constexpr auto attribute_names =
        std::forward_as_tuple("ring_size", "ring_axis", "step", "mask_type", "ring_direction");
    auto attribute_values() const {
        return std::forward_as_tuple(ring_size, ring_axis, step, mask_type, ring_direction);
    }
};

struct RingSDPAFwInputs {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    std::optional<ttnn::Tensor> preallocated_output;         // Preallocated output buffer
    std::optional<ttnn::Tensor> preallocated_intermediates;  // Preallocated intermediates buffer

    static constexpr auto attribute_names =
        std::forward_as_tuple("query_logical_shape", "query_dtype", "key_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(std::cref(query.logical_shape()), query.dtype(), std::cref(key.logical_shape()));
    }
};

using operation_attributes_t = RingSDPAFwParams;
using tensor_args_t = RingSDPAFwInputs;

using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // output, intermediates

using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::ring_sdpa_fw
