// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Helper function to pack 4-bit commands into bytes
inline void pack_command_into_buffer(
    uint8_t* route_buffer, uint16_t& byte_index, bool& is_first_nibble, uint8_t& current_packed_byte, uint8_t command) {
    if (is_first_nibble) {
        current_packed_byte = command & 0x0F;  // Lower 4 bits
        is_first_nibble = false;
    } else {
        current_packed_byte |= (command & 0x0F) << 4;  // Upper 4 bits
        route_buffer[byte_index++] = current_packed_byte;
        current_packed_byte = 0;
        is_first_nibble = true;
    }
}

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

// Device-side decoder function for 2D routing (packed paths)
template <>
inline void compressed_routing_path_t<2>::decode_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    decode_route_to_buffer_common(*this, dst_chip_id, out_route_buffer, MAX_CHIPS_LOWLAT, SINGLE_ROUTE_SIZE);
}

// Device-side decoder function for 1D routing (packed paths)
template <>
inline void compressed_routing_path_t<1>::decode_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    decode_route_to_buffer_common(*this, dst_chip_id, out_route_buffer, MAX_CHIPS_LOWLAT, SINGLE_ROUTE_SIZE);
}

// Device-side compressed decoder function for 2D routing
template <>
inline bool compressed_routing_path_t<2>::decode_compressed_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    uint32_t* route_ptr = reinterpret_cast<uint32_t*>(out_route_buffer);

    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // invalid chip
        route_ptr[0] = 0;
        return false;
    }

    // Get compressed route data
    const auto& compressed_route = compressed_paths.two[dst_chip_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();
    uint8_t ns_direction = compressed_route.get_ns_direction();
    uint8_t ew_direction = compressed_route.get_ew_direction();

    if (ns_hops == 0 && ew_hops == 0) {
        // Noop to self
        route_ptr[0] = 0;
        return false;
    }

    uint8_t total_commands = ns_hops + ew_hops;
    // Reconstruct 2D routing commands
    uint16_t byte_index = 0;
    uint8_t current_packed_byte = 0;
    bool is_first_nibble = true;

    // Phase 1: North/South routing
    if (ns_hops > 0) {
        uint8_t ns_cmd = (ns_direction == 1) ? FORWARD_SOUTH : FORWARD_NORTH;
        uint8_t ns_write_cmd = (ns_direction == 1) ? WRITE_AND_FORWARD_SOUTH : WRITE_AND_FORWARD_NORTH;

        for (uint8_t i = 0; i < ns_hops; ++i) {
            if (i == ns_hops - 1 && ew_hops == 0) {
                // Last hop and no east/west needed - write only
                pack_command_into_buffer(
                    out_route_buffer, byte_index, is_first_nibble, current_packed_byte, ns_write_cmd);
            } else {
                // Forward only
                pack_command_into_buffer(out_route_buffer, byte_index, is_first_nibble, current_packed_byte, ns_cmd);
            }
        }
    }

    // Phase 2: East/West routing
    if (ew_hops > 0) {
        uint8_t ew_cmd = (ew_direction == 1) ? FORWARD_EAST : FORWARD_WEST;
        uint8_t ew_write_cmd = (ew_direction == 1) ? WRITE_AND_FORWARD_EAST : WRITE_AND_FORWARD_WEST;

        for (uint8_t i = 0; i < ew_hops; ++i) {
            if (i == ew_hops - 1) {
                // Last hop - write only
                pack_command_into_buffer(
                    out_route_buffer, byte_index, is_first_nibble, current_packed_byte, ew_write_cmd);
            } else {
                // Forward only
                pack_command_into_buffer(out_route_buffer, byte_index, is_first_nibble, current_packed_byte, ew_cmd);
            }
        }
    }

    // If we have an odd number of hops, pack the last command with NOOP
    if (!is_first_nibble) {
        current_packed_byte |= (NOOP & 0x0F) << 4;  // Upper 4 bits as NOOP
        out_route_buffer[byte_index++] = current_packed_byte;
    }

    return true;
}

// Device-side compressed decoder function for 1D routing
template <>
inline bool compressed_routing_path_t<1>::decode_compressed_route_to_buffer(
    uint16_t dst_chip_id, uint8_t* out_route_buffer) const {
    uint32_t* route_ptr = reinterpret_cast<uint32_t*>(out_route_buffer);

    if (dst_chip_id >= MAX_CHIPS_LOWLAT) {
        // invalid chip
        *route_ptr = 0;
        return false;
    }

    const auto& compressed_route = compressed_paths.one[dst_chip_id];
    uint8_t hops = compressed_route.get_hops();
    if (hops == 0) {
        // Noop to self
        *route_ptr = 0;
        return false;
    }

    // Forward for (hops - 1) steps, then write on the final hop
    uint32_t routing_field_value =
        (FWD_ONLY_FIELD & ((1 << (hops - 1) * FIELD_WIDTH) - 1)) | (WRITE_ONLY << (hops - 1) * FIELD_WIDTH);
    *route_ptr = routing_field_value;
    return true;
}

}  // namespace tt::tt_fabric
