// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compressed_routing_path.hpp"
#include <cstring>

namespace tt::tt_fabric {

// template <uint8_t dim>
// void compressed_routing_path_t<dim>::calculate_chip_to_all_routing_fields(
//     uint16_t src_chip_id, uint16_t num_chips, uint16_t ew_dim) {
//     // Default implementation - should be specialized
//     std::memset(packed_paths, 0, sizeof(packed_paths));
// }

// 1D routing specialization
template <>
void compressed_routing_path_t<1>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id, uint16_t num_chips, uint16_t ew_dim) {
    const uint32_t FIELD_WIDTH = 2;
    const uint32_t WRITE_ONLY = 0b01;
    const uint32_t FORWARD_ONLY = 0b10;
    const uint32_t FWD_ONLY_FIELD = 0xAAAAAAAA;

    for (uint8_t dst_chip_id = 0; dst_chip_id < num_chips; ++dst_chip_id) {
        uint32_t routing_field_value = 0;

        if (src_chip_id == dst_chip_id) {
            // Noop to self
            routing_field_value = 0;
        } else {
            // Calculate distance in hops (simple linear distance)
            uint8_t distance_in_hops;
            if (src_chip_id < dst_chip_id) {
                distance_in_hops = dst_chip_id - src_chip_id;
            } else {
                distance_in_hops = src_chip_id - dst_chip_id;
            }

            // Use LowLatencyPacketHeader pattern
            // Forward for (distance_in_hops - 1) hops, then write on the final hop
            routing_field_value = (FWD_ONLY_FIELD & ((1 << (distance_in_hops - 1) * FIELD_WIDTH) - 1)) |
                                  (WRITE_ONLY << (distance_in_hops - 1) * FIELD_WIDTH);
        }

        // Store the 4-byte routing field value directly as uint32_t
        uint32_t field_offset = dst_chip_id * SINGLE_ROUTE_SIZE;
        uint32_t* route_ptr = reinterpret_cast<uint32_t*>(&packed_paths[field_offset]);
        *route_ptr = routing_field_value;
    }
}

// 2D routing specialization
template <>
void compressed_routing_path_t<2>::calculate_chip_to_all_routing_fields(
    uint16_t src_chip_id, uint16_t num_chips, uint16_t ew_dim) {
    const uint8_t NOOP = 0b0000;
    const uint8_t FORWARD_EAST = 0b0001;
    const uint8_t FORWARD_WEST = 0b0010;
    const uint8_t FORWARD_NORTH = 0b0100;
    const uint8_t FORWARD_SOUTH = 0b1000;
    const uint8_t WRITE_AND_FORWARD_EAST = 0b0001;
    const uint8_t WRITE_AND_FORWARD_WEST = 0b0010;
    const uint8_t WRITE_AND_FORWARD_NORTH = 0b0100;
    const uint8_t WRITE_AND_FORWARD_SOUTH = 0b1000;

    for (uint16_t dst_chip_id = 0; dst_chip_id < num_chips; ++dst_chip_id) {
        if (src_chip_id == dst_chip_id) {
            // Noop to self
            continue;
        }

        // Calculate 2D coordinates
        uint16_t src_col = src_chip_id / ew_dim;
        uint16_t src_row = src_chip_id % ew_dim;
        uint16_t dst_col = dst_chip_id / ew_dim;
        uint16_t dst_row = dst_chip_id % ew_dim;

        uint8_t* route_buffer = &packed_paths[dst_chip_id * SINGLE_ROUTE_SIZE];
        uint16_t hop_index = 0;
        uint16_t byte_index = 0;
        uint8_t current_packed_byte = 0;
        bool is_first_nibble = true;

        // Helper lambda to pack 4-bit commands into bytes
        auto pack_command = [&](uint8_t command) {
            if (is_first_nibble) {
                current_packed_byte = command & 0x0F;  // Lower 4 bits
                is_first_nibble = false;
            } else {
                current_packed_byte |= (command & 0x0F) << 4;  // Upper 4 bits
                route_buffer[byte_index++] = current_packed_byte;
                current_packed_byte = 0;
                is_first_nibble = true;
            }
            hop_index++;
        };

        // Phase 1: North/South routing
        if (src_col != dst_col) {
            uint16_t ns_hops = (dst_col > src_col) ? (dst_col - src_col) : (src_col - dst_col);
            uint8_t ns_direction = (dst_col > src_col) ? FORWARD_SOUTH : FORWARD_NORTH;
            uint8_t ns_write_direction = (dst_col > src_col) ? WRITE_AND_FORWARD_SOUTH : WRITE_AND_FORWARD_NORTH;

            for (uint16_t i = 0; i < ns_hops; ++i) {
                if (i == ns_hops - 1 && src_row == dst_row) {
                    // Last hop and no east/west needed - write only
                    pack_command(ns_write_direction);
                } else {
                    // Forward only
                    pack_command(ns_direction);
                }
            }
        }

        // Phase 2: East/West routing
        if (src_row != dst_row) {
            uint16_t ew_hops = (dst_row > src_row) ? (dst_row - src_row) : (src_row - dst_row);
            uint8_t ew_direction = (dst_row > src_row) ? FORWARD_EAST : FORWARD_WEST;
            uint8_t ew_write_direction = (dst_row > src_row) ? WRITE_AND_FORWARD_EAST : WRITE_AND_FORWARD_WEST;

            for (uint16_t i = 0; i < ew_hops; ++i) {
                if (i == ew_hops - 1) {
                    // Last hop - write only
                    pack_command(ew_write_direction);
                } else {
                    // Forward only
                    pack_command(ew_direction);
                }
            }
        }

        // If we have an odd number of hops, pack the last command with NOOP
        if (!is_first_nibble) {
            current_packed_byte |= (NOOP & 0x0F) << 4;  // Upper 4 bits as NOOP
            route_buffer[byte_index++] = current_packed_byte;
        }

        // Remaining bytes are already NOOP (0) from memset
    }
}

}  // namespace tt::tt_fabric
