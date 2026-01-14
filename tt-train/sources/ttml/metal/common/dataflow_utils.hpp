// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

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

// Generator the mask tile with horizontal masking.
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

// Generates a tile for broadcasting a scalar bfloat16 value.
// Only the first element of the tile is set to the scalar value (upper 16 bits of packed_scalar).
// This is used for efficient broadcast operations where only the first value is needed.
void generate_bcast_scalar_bfloat16(uint32_t cb_id, uint32_t packed_scalar) {
    cb_reserve_back(cb_id, onetile);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_id));
    ptr[0] = packed_scalar >> 16;
    cb_push_back(cb_id, onetile);
}

// Fills a tile (32x32 bfloat16 values) with a single bfloat16 value.
// This avoids writing 1024 individual 16-bit values by packing them into 512 32-bit writes.
void generate_tile_with_bfloat16_value(uint32_t cb_id, uint16_t bf16_value) {
    // Pack the same bfloat16 value into both halves of a 32-bit word
    uint32_t packed_value = (static_cast<uint32_t>(bf16_value) << 16) | bf16_value;

    generate_tile_with_packed_bfloat16_value(cb_id, packed_value);
}

// Helper template for generating matmul row reduce tile with specific type and one value.
// The tile pattern has 1.0 in the first column of even faces (left faces) and 0 elsewhere.
template <typename T, T one_value>
inline void fill_matmul_row_reduce_tile(uint32_t cb_id) {
    T* tile_ptr = reinterpret_cast<T*>(get_write_ptr(cb_id));

    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t w = 0; w < 16; ++w) {
                // Set to 'one' only in first column (w==0) of even faces (left faces)
                *tile_ptr++ = (!(face & 1U) && (w == 0)) ? one_value : static_cast<T>(0);
            }
        }
    }
}

// Generates a tile intended for performing row reduction through matrix multiplication.
// This approach is used to avoid the precision loss observed when using the reduce_tile operation.
// Automatically determines the data type from the circular buffer's data format.
// Supports: Float32, Float16_b (BF16), Int32, UInt32, UInt16, UInt8 formats.
inline void generate_matmul_row_reduce_tile(uint32_t cb_id) {
    const DataFormat data_format = get_dataformat(cb_id);

    cb_reserve_back(cb_id, onetile);

    switch (data_format) {
        case DataFormat::Float32:
            fill_matmul_row_reduce_tile<uint32_t, 0x3F800000>(cb_id);  // fp32 1.0
            break;
        case DataFormat::Int32:
        case DataFormat::UInt32: fill_matmul_row_reduce_tile<uint32_t, 1>(cb_id); break;
        case DataFormat::UInt16: fill_matmul_row_reduce_tile<uint16_t, 1>(cb_id); break;
        case DataFormat::UInt8: fill_matmul_row_reduce_tile<uint8_t, 1>(cb_id); break;
        default:                                                   // Float16_b and other bf16 variants
            fill_matmul_row_reduce_tile<uint16_t, 0x3F80>(cb_id);  // bf16 1.0
            break;
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

// ----- Dataflow tile transfer utilities -----

/**
 * Utility: read a single tile from DRAM to CB.
 *
 * @param cb_idx Circular buffer index to write to
 * @param addr_gen Address generator for DRAM access
 * @param tile_idx Tile index in DRAM
 */
template <typename AddrGen>
inline void read_one_tile(const uint32_t cb_idx, const AddrGen& addr_gen, const uint32_t tile_idx) {
    cb_reserve_back(cb_idx, onetile);
    const uint32_t l1_addr = get_write_ptr(cb_idx);
    noc_async_read_page(tile_idx, addr_gen, l1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_idx, onetile);
}

/**
 * Utility: read contiguous tiles in row-major order from DRAM to CB.
 *
 * @param cb_idx Circular buffer index to write to
 * @param addr_gen Address generator for DRAM access
 * @param start_idx Starting tile index in DRAM
 * @param num_tiles_to_read Number of tiles to actually read (may be less than num_tiles_to_push for tail blocks)
 * @param tile_bytes Size of each tile in bytes
 * @param num_tiles_to_push Number of tiles to reserve/push in CB (buffer capacity)
 * @param UseBarrier Whether to call noc_async_read_barrier() (compile-time)
 */
template <bool UseBarrier = true, typename AddrGen>
inline void read_tiles_by_row(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t start_idx,
    const uint32_t num_tiles_to_read,
    const uint32_t tile_bytes,
    const uint32_t num_tiles_to_push) {
    cb_reserve_back(cb_idx, num_tiles_to_push);
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles_to_read; ++t) {
        noc_async_read_page(start_idx + t, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
    // Note: If UseBarrier is false, caller must ensure noc_async_read_barrier() is called later as well as
    // cb_push_back()
    if constexpr (UseBarrier) {
        noc_async_read_barrier();
        cb_push_back(cb_idx, num_tiles_to_push);
    }
}

/**
 * Utility: read contiguous tiles in column-major order from DRAM to CB.
 *
 * @param cb_idx Circular buffer index to write to
 * @param addr_gen Address generator for DRAM access
 * @param start_idx Starting tile index in DRAM
 * @param num_tiles_to_read Number of tiles to actually read (may be less than num_tiles_to_push for tail blocks)
 * @param tile_bytes Size of each tile in bytes
 * @param stride Stride between consecutive tiles in column-major order
 * @param num_tiles_to_push Number of tiles to reserve/push in CB (buffer capacity)
 * @param UseBarrier Whether to call noc_async_read_barrier() (compile-time)
 */
template <bool UseBarrier = true, typename AddrGen>
inline void read_tiles_by_col(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t start_idx,
    const uint32_t num_tiles_to_read,
    const uint32_t tile_bytes,
    const uint32_t stride,
    const uint32_t num_tiles_to_push) {
    cb_reserve_back(cb_idx, num_tiles_to_push);
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles_to_read; ++t) {
        uint32_t tile_idx = start_idx + t * stride;
        noc_async_read_page(tile_idx, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
    // Note: If UseBarrier is false, caller must ensure noc_async_read_barrier() is called later as well as
    // cb_push_back()
    if constexpr (UseBarrier) {
        noc_async_read_barrier();
        cb_push_back(cb_idx, num_tiles_to_push);
    }
}

/**
 * Utility: write a block of tiles from CB to DRAM in row-major order.
 *
 * @param cb_idx Circular buffer index to read from
 * @param addr_gen Address generator for DRAM access
 * @param start_idx Starting tile index in DRAM
 * @param num_tiles_to_write Number of tiles to actually write (may be less than num_tiles_to_pop for tail blocks)
 * @param tile_bytes Size of each tile in bytes
 * @param num_tiles_to_pop Number of tiles to wait/pop from CB (buffer capacity)
 * @param UseBarrier Whether to call noc_async_write_barrier() (compile-time)
 */
template <bool UseBarrier = true, typename AddrGen>
inline void write_tiles_by_row(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t start_idx,
    const uint32_t num_tiles_to_write,
    const uint32_t tile_bytes,
    const uint32_t num_tiles_to_pop) {
    cb_wait_front(cb_idx, num_tiles_to_pop);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles_to_write; ++t) {
        noc_async_write_page(start_idx + t, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
    // Note: If UseBarrier is false, caller must ensure noc_async_write_barrier() is called later as well as
    // cb_pop_front()
    if constexpr (UseBarrier) {
        noc_async_write_barrier();
        cb_pop_front(cb_idx, num_tiles_to_pop);
    }
}

// ----- Higher-level utility functions -----

// Read a full row of tiles by blocks, commonly used pattern across kernels
template <typename AddrGen>
inline void read_full_row_tiles(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t tile_bytes,
    const uint32_t row_start_idx) {
    for (uint32_t j = 0; j < Wt; j += block_size) {
        uint32_t current_block_size = (j + block_size <= Wt) ? block_size : (Wt - j);
        read_tiles_by_row(cb_idx, addr_gen, row_start_idx + j, current_block_size, tile_bytes, block_size);
    }
}

// Write a full row of tiles by blocks, commonly used pattern across kernels
template <typename AddrGen>
inline void write_full_row_tiles(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t tile_bytes,
    const uint32_t row_start_idx) {
    for (uint32_t j = 0; j < Wt; j += block_size) {
        uint32_t current_block_size = (j + block_size <= Wt) ? block_size : (Wt - j);
        write_tiles_by_row(cb_idx, addr_gen, row_start_idx + j, current_block_size, tile_bytes, block_size);
    }
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
