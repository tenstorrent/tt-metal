// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_fw.hpp"

#include "device/ring_sdpa_fw_device_operation.hpp"

namespace ttml::metal {

void ring_sdpa_fw(
    std::vector<RingSDPAFwChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_sdpa_fw::RingDirection ring_direction,
    ops::RingLoadBalancing load_balancing) {
    ttnn::prim::ttml_ring_sdpa_fw(
        std::move(chunks), ring_size, ring_axis, step, mask_type, ring_direction, load_balancing);
}

}  // namespace ttml::metal
