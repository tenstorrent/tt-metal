// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

using RingSDPAFwChunk = ops::ring_sdpa_fw::RingSDPAFwChunk;

// Each device writes only the outputs of the chunk get_ring_step_work assigns it; other outputs keep their contents.
void ring_sdpa_fw(
    std::vector<RingSDPAFwChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_sdpa_fw::RingDirection ring_direction = ops::ring_sdpa_fw::RingDirection::Backward,
    ops::RingLoadBalancing load_balancing = ops::RingLoadBalancing::None);

}  // namespace ttml::metal
