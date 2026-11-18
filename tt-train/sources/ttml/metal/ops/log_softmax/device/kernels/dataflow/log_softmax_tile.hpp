// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/bfloat16.h"

// Element access inside one tile in L1 (faces in row-major order, row-major within a face), for the few
// scalar edits the data-movement kernels make: the selected element of each row and column-vector statistics.
// Constant regions (tile padding, constant tiles) are copied with the NoC instead. BF16 tiles hold the high half
// of the FP32 bit pattern. The host defines the geometry from the input's tensor_spec().tile().
namespace log_softmax_tile {
constexpr uint32_t kTileHeight = LOG_SOFTMAX_TILE_HEIGHT, kTileWidth = LOG_SOFTMAX_TILE_WIDTH;
constexpr uint32_t kFaceHeight = LOG_SOFTMAX_FACE_HEIGHT, kFaceWidth = LOG_SOFTMAX_FACE_WIDTH;
// One unit's partial row statistics as exchanged through DRAM: up to four column vectors (one per
// statistic, kTileHeight values each) packed contiguously, rather than one tile per statistic.
constexpr uint32_t kPartialStatistics = 4;
constexpr uint32_t kPartialWords = kPartialStatistics * kTileHeight;
constexpr uint32_t kPartialBytes = kPartialWords * sizeof(uint32_t);
constexpr uint32_t offset(uint32_t r, uint32_t c) {
    return ((r / kFaceHeight) * (kTileWidth / kFaceWidth) + c / kFaceWidth) * (kFaceHeight * kFaceWidth) +
           (r % kFaceHeight) * kFaceWidth + c % kFaceWidth;
}
inline volatile tt_l1_ptr uint32_t* words(uint32_t base) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
}
inline volatile tt_l1_ptr uint16_t* halves(uint32_t base) {
    return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
}
// FP32 bit pattern of element `at`.
template <uint32_t fp32>
inline uint32_t load(uint32_t base, uint32_t at) {
    if constexpr (fp32) {
        return words(base)[at];
    } else {
        return static_cast<uint32_t>(halves(base)[at]) << 16;
    }
}
// Stores FP32 bits, rounding to nearest even for BF16 tiles.
template <uint32_t fp32>
inline void store(uint32_t base, uint32_t at, uint32_t bits) {
    if constexpr (fp32) {
        words(base)[at] = bits;
    } else {
        halves(base)[at] = fp32_to_bf16(__builtin_bit_cast(float, bits));
    }
}
inline void store_fp32(uint32_t base, uint32_t at, uint32_t bits) {
    words(base)[at] = bits;
}
// Fills a tile of `bytes` at `base` with one constant (FP32 `bits`, or their high half for a BF16 tile): stores the
// first 16 bytes, then doubles the filled prefix with loopback NoC reads. Done once per kernel, for the constant
// tiles that copy_columns and the per-row fills read from.
template <uint32_t fp32>
inline void fill_constant_tile(uint32_t base, uint32_t bytes, uint32_t bits) {
    constexpr uint32_t kPrefixBytes = 16;
    for (uint32_t i = 0; i < kPrefixBytes / (fp32 ? sizeof(uint32_t) : sizeof(uint16_t)); ++i) {
        store<fp32>(base, i, bits);
    }
    for (uint32_t filled = kPrefixBytes; filled < bytes; filled *= 2) {
        noc_async_read(get_noc_addr(base), base + filled, filled < bytes - filled ? filled : bytes - filled);
        noc_async_read_barrier();
    }
}
// Copies columns [first, kTileWidth) of every row from the tile at `source` to the tile at `destination`, both in
// the same format with `element_bytes` per element: one NoC read per face that lies wholly in those columns, one
// per face row otherwise. Both tiles start 16-byte aligned, so source and destination share their alignment, as
// the NoC requires. The caller waits with noc_async_read_barrier().
template <uint32_t element_bytes>
inline void copy_columns(uint32_t source, uint32_t destination, uint32_t first) {
    const uint64_t source_noc = get_noc_addr(source);
    for (uint32_t face_row = 0; face_row < kTileHeight / kFaceHeight; ++face_row) {
        for (uint32_t face_column = 0; face_column < kTileWidth / kFaceWidth; ++face_column) {
            const uint32_t lo = face_column * kFaceWidth, hi = lo + kFaceWidth;
            if (hi <= first) {
                continue;
            }
            if (lo >= first) {
                const uint32_t at = offset(face_row * kFaceHeight, lo) * element_bytes;
                noc_async_read(source_noc + at, destination + at, kFaceHeight * kFaceWidth * element_bytes);
                continue;
            }
            for (uint32_t r = 0; r < kFaceHeight; ++r) {
                const uint32_t at = offset(face_row * kFaceHeight + r, first) * element_bytes;
                noc_async_read(source_noc + at, destination + at, (hi - first) * element_bytes);
            }
        }
    }
}
inline void copy_fp32(uint32_t from, uint32_t at_from, uint32_t to, uint32_t at_to) {
    words(to)[at_to] = words(from)[at_from];
}
// Selected maximum index of each row, stored as an exact FP32 integer in column 0.
inline void read_pivots(uint32_t base, uint32_t (&pivot)[kTileHeight]) {
    for (uint32_t r = 0; r < kTileHeight; ++r) {
        const uint32_t bits = words(base)[offset(r, 0)];
        const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127;
        if (bits & 0x80000000U || exponent < 0 || exponent > 23) {
            pivot[r] = exponent < 0 && !(bits & 0x80000000U) ? 0U : 0xFFFFFFFFU;  // Zero, or invalid (padding).
            continue;
        }
        pivot[r] = ((bits & 0x7FFFFFU) | 0x800000U) >> (23 - exponent);
    }
}
}  // namespace log_softmax_tile
