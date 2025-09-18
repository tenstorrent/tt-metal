// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Device-side decoder function for 1D routing (packed paths)
template <>
inline bool routing_path_t<1, false>::decode_route_to_buffer(
    uint16_t dst_chip_id, volatile uint8_t* out_route_buffer) const {
    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // Out of bounds - fill buffer with NOOPs/zeros
        for (uint16_t i = 0; i < SINGLE_ROUTE_SIZE; ++i) {
            out_route_buffer[i] = 0;
        }
        ASSERT(false);  // catched only watcher enabled. Otherwise make behavior consistent as returning false.
        return false;
    }

    const uint8_t* packed_route = &this->paths[dst_chip_id * SINGLE_ROUTE_SIZE];
    // Copy packed data directly to output buffer
    for (uint16_t i = 0; i < SINGLE_ROUTE_SIZE; ++i) {
        out_route_buffer[i] = packed_route[i];
    }
    return true;
}

// Device-side compressed decoder function for 2D routing
template <>
inline bool routing_path_t<2, true>::decode_route_to_buffer(
    uint16_t dst_chip_id, volatile uint8_t* out_route_buffer) const {
    auto route_ptr = reinterpret_cast<volatile uint32_t*>(out_route_buffer);

    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // invalid chip
        route_ptr[0] = 0;
        ASSERT(false);  // catched only watcher enabled. Otherwise make behavior consistent as returning false.
        return false;
    }

    // Get compressed route data
    const auto& compressed_route = paths[dst_chip_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();
    uint8_t ns_direction = compressed_route.get_ns_direction();
    uint8_t ew_direction = compressed_route.get_ew_direction();

    if (ns_hops == 0 && ew_hops == 0) {
        // Noop to self
        route_ptr[0] = 0;
        return false;
    }

    // Reconstruct 2D routing commands (1 command per byte)
    uint8_t byte_index = 0;
    // Construct route to match fabric_set_unicast_route encoding:
    // - Final hop uses the opposite-direction bit (no forward) to indicate write
    // - If both NS and EW exist: emit (ns_hops - 1) NS forwards, then ew_hops EW forwards, then 1 EW write (opposite)
    // - If only NS: emit (ns_hops - 1) NS forwards, then 1 NS write (opposite)
    // - If only EW: emit (ew_hops - 1) EW forwards, then 1 EW write (opposite)

    // Determine forward and write(opposite) commands per dimension
    const uint8_t ns_forward_cmd = (ns_direction == 1) ? FORWARD_SOUTH : FORWARD_NORTH;
    const uint8_t ns_write_cmd_opposite = (ns_direction == 1) ? FORWARD_NORTH : FORWARD_SOUTH;  // opposite

    const uint8_t ew_forward_cmd = (ew_direction == 1) ? FORWARD_EAST : FORWARD_WEST;
    const uint8_t ew_write_cmd_opposite = (ew_direction == 1) ? FORWARD_WEST : FORWARD_EAST;  // opposite

    if (ns_hops > 0 && ew_hops > 0) {
        // NS forwards for ns_hops - 1
        for (uint8_t i = 0; i < ns_hops - 1; ++i) {
            out_route_buffer[byte_index++] = ns_forward_cmd;
        }
        // EW forwards for ew_hops
        for (uint8_t i = 0; i < ew_hops; ++i) {
            out_route_buffer[byte_index++] = ew_forward_cmd;
        }
        // Final write in EW with opposite direction bit
        out_route_buffer[byte_index++] = ew_write_cmd_opposite;
    } else if (ns_hops > 0) {
        // Only NS path: (ns_hops - 1) forwards + 1 write(opposite)
        for (uint8_t i = 0; i < ns_hops - 1; ++i) {
            out_route_buffer[byte_index++] = ns_forward_cmd;
        }
        out_route_buffer[byte_index++] = ns_write_cmd_opposite;
    } else if (ew_hops > 0) {
        // Only EW path: (ew_hops - 1) forwards + 1 write(opposite)
        for (uint8_t i = 0; i < ew_hops - 1; ++i) {
            out_route_buffer[byte_index++] = ew_forward_cmd;
        }
        out_route_buffer[byte_index++] = ew_write_cmd_opposite;
    }

    out_route_buffer[byte_index] = NOOP;

    return true;
}

// Device-side compressed decoder function for 1D routing
template <>
inline bool routing_path_t<1, true>::decode_route_to_buffer(uint16_t hops, volatile uint8_t* out_route_buffer) const {
    return true;
}

inline bool decode_route_to_buffer_by_hops(uint16_t hops, volatile uint8_t* out_route_buffer) {
    auto route_ptr = reinterpret_cast<volatile uint32_t*>(out_route_buffer);

    if (hops >= routing_path_t<1, true>::MAX_CHIPS_LOWLAT || hops == 0) {
        // invalid chip or Noop to self
        *route_ptr = 0;
        ASSERT(false);  // catched only watcher enabled. Otherwise make behavior consistent as returning false.
        return false;
    }

    // Forward for (hops - 1) steps, then write on the final hop
    uint32_t routing_field_value =
        (routing_path_t<1, true>::FWD_ONLY_FIELD & ((1 << (hops - 1) * routing_path_t<1, true>::FIELD_WIDTH) - 1)) |
        (routing_path_t<1, true>::WRITE_ONLY << (hops - 1) * routing_path_t<1, true>::FIELD_WIDTH);
    *route_ptr = routing_field_value;
    return true;
}

// Device-side compressed decoder function for 1D routing
inline bool decode_route_to_buffer_by_dev(uint16_t dst_chip_id, volatile uint8_t* out_route_buffer) {
    tt_l1_ptr tensix_routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr tensix_routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    uint16_t my_device_id = routing_table->my_device_id;
    uint16_t hops = my_device_id > dst_chip_id ? my_device_id - dst_chip_id : dst_chip_id - my_device_id;

    return decode_route_to_buffer_by_hops(hops, out_route_buffer);
}

}  // namespace tt::tt_fabric
