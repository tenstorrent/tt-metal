// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>
#include <cstdint>
#include <cstring>

// Given a source array of 32-bit elements that contains a densely-packed array of elements of another size,
// extract the packed elements and store each element in the destination array.
// Note: the function does not check for overflow in the either the source or destination array. The caller
//       must ensure that the arrays are large enough.
//
// src_array         - Source array containing the packed elements
// src_element_bits  - Number of bits in each pakced element (must be LESS than 32)
// dest_array        - Destination array to store the extracted bits
// num_dest_elements - Number of elements in the destination array
void extract_bit_array(uint32_t* src_array, int src_element_bits, uint32_t* dest_array, int num_dest_elements) {
    int bits_processed = 0;      // Tracks the number of bits processed in the current src_array element
    int src_index = 0;           // Index for the current source element being processed
    uint32_t current_value = 0;  // Temporary storage for the value being extracted

    for (int i = 0; i < num_dest_elements; i++) {
        current_value = 0;  // Reset current value for each destination element

        int bits_to_process = src_element_bits;  // Bits left to process for the current destination element
        while (bits_to_process > 0) {
            int bits_available = 32 - bits_processed;  // Bits available in the current src_array element
            int bits_to_take = bits_to_process < bits_available
                                   ? bits_to_process
                                   : bits_available;  // Bits to take from the current src_array element

            // Extract the bits
            uint32_t mask = (1 << bits_to_take) - 1u;  // Mask to extract_bit_array the bits
            current_value |= ((src_array[src_index] >> bits_processed) & mask) << (src_element_bits - bits_to_process);

            bits_processed += bits_to_take;
            bits_to_process -= bits_to_take;

            if (bits_processed >=
                32) {  // If we've processed all bits in the current src_array element, move to the next
                src_index++;
                bits_processed = 0;
            }
        }

        // Store the extracted value in the destination array
        dest_array[i] = current_value;
    }
}

// Given a source array of elements that only use the 'src_element_bits' least significant bits,
// pack the elements into a destination array of 32-bit elements in a densely-packed manner, where
// the least significant bits of the destination array elements are filled first and all 32 bits are used.
void pack_bit_array(uint32_t* src_array, int src_element_bits, uint32_t* dest_array, int num_src_elements) {
    int dest_index = 0;   // Index for the current destination array element being filled
    int bits_filled = 0;  // Tracks the number of bits filled in the current dest_array element

    auto dest_bytes = num_src_elements * src_element_bits / CHAR_BIT;
    memset(dest_array, 0, dest_bytes);

    for (int i = 0; i < num_src_elements; i++) {
        uint32_t current_value = src_array[i];  // Current source element to pack
        int bits_to_pack = src_element_bits;    // Number of bits to pack from the current source element

        while (bits_to_pack > 0) {
            int bits_available = 32 - bits_filled;  // Space available in the current dest_array element
            int bits_to_write = bits_to_pack < bits_available ? bits_to_pack : bits_available;  // Bits to write now

            // Pack the bits
            dest_array[dest_index] |= (current_value & ((1 << bits_to_write) - 1)) << bits_filled;

            current_value >>= bits_to_write;  // Remove packed bits from current value
            bits_filled += bits_to_write;     // Update bits filled in the current dest_array element
            bits_to_pack -= bits_to_write;    // Decrease remaining bits to pack

            if (bits_filled == 32) {  // If the current dest_array element is full, move to the next
                dest_index++;
                bits_filled = 0;  // Reset bits filled counter
            }
        }
    }
}
