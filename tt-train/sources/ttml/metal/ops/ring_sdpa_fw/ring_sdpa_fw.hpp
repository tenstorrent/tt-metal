// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns [output, intermediates] of lane 0. With RingLaneSelection::ZigZag the devices
// assigned lane 1 at this step write into lane1->output / lane1->intermediates instead
// (see RingLaneSelection in metal/ops/common/ring_sdpa_utils.hpp).
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type = AttentionMaskType::None,
    ops::ring_sdpa_fw::RingDirection ring_direction = ops::ring_sdpa_fw::RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt,
    ops::RingLaneSelection lane_selection = ops::RingLaneSelection::None,
    const std::optional<ops::ring_sdpa_fw::RingSDPAFwLane>& lane1 = std::nullopt);

}  // namespace ttml::metal
