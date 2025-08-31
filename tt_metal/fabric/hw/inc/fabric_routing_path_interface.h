// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Common helper function for both 1D and 2D routing
template <uint8_t dim>
inline void decode_route_to_buffer_common(
    const compressed_routing_path_t<dim>& routing_path,
    uint16_t dst_chip_id,
    uint8_t* out_route_buffer,
    uint16_t max_chips,
    uint16_t route_size) {
    if (dst_chip_id >= max_chips) {
        // Out of bounds - fill buffer with NOOPs/zeros
        for (uint16_t i = 0; i < route_size; ++i) {
            out_route_buffer[i] = 0;
        }
        return;
    }

    const uint8_t* packed_route = &routing_path.packed_paths[dst_chip_id * route_size];
    // Copy packed data directly to output buffer
    for (uint16_t i = 0; i < route_size; ++i) {
        out_route_buffer[i] = packed_route[i];
    }
}

// Device-side decoder function for 2D routing
template <>
inline void compressed_routing_path_t<2>::decode_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    decode_route_to_buffer_common(*this, dst_chip_id, out_route_buffer, MAX_CHIPS_LOWLAT, SINGLE_ROUTE_SIZE);
}

// Device-side decoder function for 1D routing
template <>
inline void compressed_routing_path_t<1>::decode_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    decode_route_to_buffer_common(*this, dst_chip_id, out_route_buffer, MAX_CHIPS_LOWLAT, SINGLE_ROUTE_SIZE);
}

}  // namespace tt::tt_fabric
