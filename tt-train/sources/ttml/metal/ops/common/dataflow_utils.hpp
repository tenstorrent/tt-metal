// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

constexpr uint32_t TILE_WIDTH = 32U;
constexpr uint32_t TILE_HEIGHT = 32U;
constexpr uint32_t FACE_WIDTH = 16U;
constexpr uint32_t FACE_HEIGHT = 16U;
constexpr uint32_t onetile = 1U;

inline uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    // Get local coordinates within the tile
    uint32_t local_row = h % TILE_HEIGHT;
    uint32_t local_col = w % TILE_WIDTH;

    // Determine the index offset based on which quadrant we're in
    uint32_t offset = 0;

    // If we're in the right half (columns beyond FACE_WIDTH)
    if (local_col >= FACE_WIDTH) {
        local_col -= FACE_WIDTH;
        offset += FACE_HEIGHT * FACE_WIDTH;  // Right face offset
    }

    // If we're in the bottom half (rows beyond FACE_WIDTH)
    if (local_row >= FACE_WIDTH) {
        local_row -= FACE_WIDTH;
        offset += FACE_HEIGHT * TILE_WIDTH;  // Bottom face offset
    }

    // Final index within the tile
    uint32_t index = offset + local_row * FACE_WIDTH + local_col;
    return index;
}

inline std::pair<uint32_t, uint32_t> get_page_and_offset(uint32_t tiled_row, uint32_t tiled_H) {
    uint32_t page = tiled_row / tiled_H;
    uint32_t offset = (tiled_row % tiled_H) * 32U * sizeof(uint32_t);
    return {page, offset};
}

// ----- Tile generation functions -----

// Generater the mask tile with horizontal masking.
// Each tile face is 16x16, and there are 4 faces per tile.
void generate_mask_tile(uint32_t cb_id, uint16_t fill_value, uint16_t mask_fill_value, uint32_t mask_width) {
    cb_reserve_back(cb_id, onetile);

    uint16_t* tile_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));

    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t face_offset = (face & 1U) << 4U;
        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t w = 0; w < 16; ++w) {
                *tile_ptr++ = (face_offset + w < mask_width) ? fill_value : mask_fill_value;
            }
        }
    }

    cb_push_back(cb_id, onetile);
}

// Fills a tile (32x32 bfloat16 values) with a packed 32-bit value,
// where each 32-bit word contains two identical bfloat16 values.
// This improves performance by writing 512 uint32_t values instead of 1024 uint16_t values.
// The packed data is written into the circular buffer `cb_id`.
void generate_tile_with_packed_bfloat16_value(uint32_t cb_id, uint32_t packed_bf16_value) {
    cb_reserve_back(cb_id, onetile);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_id));
    // 512 = 32x16
    for (uint32_t i = 0; i < 512U; ++i) {
        *ptr++ = packed_bf16_value;
    }
    cb_push_back(cb_id, onetile);
}

// Fills a tile (32x32 bfloat16 values) with a single bfloat16 value.
// This avoids writing 1024 individual 16-bit values by packing them into 512 32-bit writes.
void generate_tile_with_bfloat16_value(uint32_t cb_id, uint16_t bf16_value) {
    // Pack the same bfloat16 value into both halves of a 32-bit word
    uint32_t packed_value = (static_cast<uint32_t>(bf16_value) << 16) | bf16_value;

    generate_tile_with_packed_bfloat16_value(cb_id, packed_value);
}

// Generates a tile intended for performing row reduction through matrix multiplication.
// This approach is used to avoid the precision loss observed when using the reduce_tile operation.
void generate_matmul_row_reduce_tile(uint32_t cb_id) {
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;

    cb_reserve_back(cb_id, onetile);
    uint16_t* tile_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));

    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t offset = (face & 1U) << 4U;
        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t w = 0; w < 16; ++w) {
                if (!(face & 1U) && (w == 0)) {  // check whether face is even and width is zero
                    *tile_ptr++ = one;
                } else {
                    *tile_ptr++ = zero;
                }
            }
        }
    }
    cb_push_back(cb_id, onetile);
}

// ----- Type conversion helper functions -----
// These functions provide bitwise conversions between float, uint32_t, and bfloat16.
// We use them instead of std::bit_cast because the kernel code is compiled with C++17,
// which does not support std::bit_cast (introduced in C++20).

// Converts a bfloat16 (stored in the lower 16 bits) to a float.
// This is done by shifting the bfloat16 to the upper 16 bits of a 32-bit integer
// and reinterpreting it as a float using memcpy.
inline float bfloat16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

// Converts a float to bfloat16 by extracting the upper 16 bits
// of the float's 32-bit binary representation.
inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

// Converts a uint32_t bit pattern to a float (bitwise reinterpretation)
inline float uint32_to_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

// ----- Printing helper functions -----

void print_tile(uint32_t cb_idx, uint32_t tile_idx, bool untilize = false) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}
