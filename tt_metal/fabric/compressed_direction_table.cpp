// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compressed_direction_table.hpp"

namespace tt::tt_fabric {

template <std::uint32_t ArraySize>
void direction_table_t<ArraySize>::set_direction(std::uint16_t index, std::uint8_t direction) {
    std::uint32_t bit_index = index * BITS_PER_COMPRESSED_ENTRY;
    std::uint32_t byte_index = bit_index / BITS_PER_BYTE;
    std::uint32_t bit_offset = bit_index % BITS_PER_BYTE;

    if (bit_offset <= BITS_PER_BYTE - BITS_PER_COMPRESSED_ENTRY) {
        // All 3 bits are in the same byte
        packed_directions[byte_index] &= ~(COMPRESSED_ENTRY_MASK << bit_offset);             // Clear bits
        packed_directions[byte_index] |= (direction & COMPRESSED_ENTRY_MASK) << bit_offset;  // Set bits
    } else {
        // Bits span across two bytes
        std::uint8_t bits_in_first_byte = BITS_PER_BYTE - bit_offset;
        std::uint8_t bits_in_second_byte = BITS_PER_COMPRESSED_ENTRY - bits_in_first_byte;

        // Clear and set bits in first byte
        packed_directions[byte_index] &= ~(((1 << bits_in_first_byte) - 1) << bit_offset);
        packed_directions[byte_index] |= (direction & ((1 << bits_in_first_byte) - 1)) << bit_offset;

        // Clear and set bits in second byte
        packed_directions[byte_index + 1] &= ~((1 << bits_in_second_byte) - 1);
        packed_directions[byte_index + 1] |= (direction >> bits_in_first_byte) & ((1 << bits_in_second_byte) - 1);
    }
}

template <std::uint32_t ArraySize>
std::uint8_t direction_table_t<ArraySize>::compress_value(std::uint8_t original_value) const {
    switch (original_value) {
        case eth_chan_directions::EAST: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_EAST);
        case eth_chan_directions::WEST: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_WEST);
        case eth_chan_directions::NORTH: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_NORTH);
        case eth_chan_directions::SOUTH: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_SOUTH);
        case eth_chan_directions::Z: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_Z);
        case eth_chan_magic_values::INVALID_DIRECTION:
            return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_INVALID_DIRECTION);
        case eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY:
        default: return static_cast<std::uint8_t>(compressed_routing_values::COMPRESSED_INVALID_ROUTING_TABLE_ENTRY);
    }
}

template <std::uint32_t ArraySize>
void direction_table_t<ArraySize>::set_original_direction(std::uint16_t index, std::uint8_t original_direction) {
    set_direction(index, compress_value(original_direction));
}

// Explicit instantiations for routing_l1_info_t
template struct direction_table_t<MAX_MESH_SIZE>;
template struct direction_table_t<MAX_NUM_MESHES>;

}  // namespace tt::tt_fabric
