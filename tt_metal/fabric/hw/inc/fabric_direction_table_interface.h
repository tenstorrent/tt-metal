// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Device-side (HW) inline implementations for direction_table_t
// Host-side functionality is implemented in tt_metal/fabric/compressed_direction_table.cpp

template <std::uint32_t ArraySize>
inline std::uint8_t direction_table_t<ArraySize>::get_direction(std::uint16_t index) const {
    std::uint32_t bit_index = index * BITS_PER_COMPRESSED_ENTRY;
    std::uint32_t byte_index = bit_index / BITS_PER_BYTE;
    std::uint32_t bit_offset = bit_index % BITS_PER_BYTE;

    if (bit_offset <= BITS_PER_BYTE - BITS_PER_COMPRESSED_ENTRY) {
        // All 3 bits are in the same byte
        return (packed_directions[byte_index] >> bit_offset) & COMPRESSED_ENTRY_MASK;
    } else {
        // Bits span across two bytes
        std::uint8_t low_bits =
            (packed_directions[byte_index] >> bit_offset) & ((1 << (BITS_PER_BYTE - bit_offset)) - 1);
        std::uint8_t high_bits = (packed_directions[byte_index + 1] &
                                  ((1 << (BITS_PER_COMPRESSED_ENTRY - (BITS_PER_BYTE - bit_offset))) - 1))
                                 << (BITS_PER_BYTE - bit_offset);
        return low_bits | high_bits;
    }
}

template <std::uint32_t ArraySize>
inline std::uint8_t direction_table_t<ArraySize>::decompress_value(std::uint8_t compressed_value) const {
    switch (static_cast<compressed_routing_values>(compressed_value)) {
        case compressed_routing_values::COMPRESSED_EAST: return eth_chan_directions::EAST;
        case compressed_routing_values::COMPRESSED_WEST: return eth_chan_directions::WEST;
        case compressed_routing_values::COMPRESSED_NORTH: return eth_chan_directions::NORTH;
        case compressed_routing_values::COMPRESSED_SOUTH: return eth_chan_directions::SOUTH;
        case compressed_routing_values::COMPRESSED_Z: return eth_chan_directions::Z;
        case compressed_routing_values::COMPRESSED_INVALID_DIRECTION: return eth_chan_magic_values::INVALID_DIRECTION;
        case compressed_routing_values::COMPRESSED_INVALID_ROUTING_TABLE_ENTRY:
            return eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY;
        default: return eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY;
    }
}

template <std::uint32_t ArraySize>
inline std::uint8_t direction_table_t<ArraySize>::get_original_direction(std::uint16_t index) const {
    return decompress_value(get_direction(index));
}

}  // namespace tt::tt_fabric
