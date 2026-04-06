// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

using RingDirection = ttml::ttnn_fixed::distributed::RingShiftDirection;
using ttml::metal::AttentionMaskType;
// Determine if device should execute at this step and which mask type to use
// Returns: (should_execute, mask_type_to_use)
inline std::pair<bool, AttentionMaskType> get_device_execution_info(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    RingDirection ring_direction) {
    if (mask_type != AttentionMaskType::Causal) {
        // Non-causal: all devices execute with no mask (full attention)
        return {true, AttentionMaskType::None};
    }

    // Causal masking logic for ring attention:
    // At step s, device d processes K/V from source device based on ring direction
    // - Backward direction: src = (d + s) % ring_size
    // - Forward direction: src = (d - s + ring_size) % ring_size
    // Then apply causal logic:
    // - If src == device_id: diagonal chunk, use causal mask
    // - If src < device_id: earlier chunk, use full attention (no mask)
    // - If src > device_id: later chunk, skip (all positions masked)
    uint32_t src_device;
    if (ring_direction == RingDirection::Backward) {
        src_device = (device_ring_id + step) % ring_size;
    } else {
        src_device = (device_ring_id - step + ring_size) % ring_size;
    }

    if (src_device == device_ring_id) {
        return {true, AttentionMaskType::Causal};  // Diagonal: use causal mask
    } else if (src_device < device_ring_id) {
        return {true, AttentionMaskType::None};  // Earlier: full attention (no mask)
    } else {
        return {false, AttentionMaskType::None};  // Later: skip
    }
}
