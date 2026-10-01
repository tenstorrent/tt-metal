// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef ARCH_BLACKHOLE
#include "ckernel_debug.h"
#endif

namespace ckernel {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Reads a single tile from the DEST register through the RISC-V memory-mapped
 * dest register and copies it into the provided buffer.
 *
 * Supported data formats: Float32, Float16, Float16_b, Int32, UInt32, UInt16,
 * Int8, UInt8.
 *
 * Return value: None
 *
 * | Template Param  | Description                                                                                  | Type       | Valid Range                                   | Required |
 * |-----------------|----------------------------------------------------------------------------------------------|------------|-----------------------------------------------|----------|
 * | fmt             | Data format of the values stored in DEST                                                     | DataFormat | See supported formats above                   | True     |
 *
 * | Argument        | Description                                                                                  | Type       | Valid Range                                   | Required |
 * |-----------------|----------------------------------------------------------------------------------------------|------------|-----------------------------------------------|----------|
 * | tile_id         | Index of the tile within DEST to read                                           | uint32_t   | Must be less than the size of the DST register buffer      | True     |
 * | dst_buffer      | Destination buffer with space for TILE_HEIGHT * TILE_WIDTH elements of *fmt*    | void*      | Non-null                                                   | True     |
 * | enable_swizzle  | When true, read with hardware swizzling enabled                                 | bool       | true or false                                              | False    |
 */
// clang-format on
template <DataFormat fmt>
ALWI void dbg_read_dest_tile(uint32_t tile_id, void* dst_buffer, bool enable_swizzle = true) {
    static_assert(dbg_dest_fmt_supported(fmt), "dbg_read_dest_tile: unsupported DataFormat");
    MATH((dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(fmt, tile_id, dst_buffer, enable_swizzle)));
}

// clang-format off
/**
 * Writes a single tile into the DEST register through the RISC-V memory-mapped
 * dest register.
 *
 * Supported data formats: Float32, Float16, Float16_b, Int32, UInt32, UInt16,
 * Int8, UInt8.
 *
 * Return value: None
 *
 * | Template Param  | Description                                                                                  | Type        | Valid Range                                                | Required |
 * |-----------------|----------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------|----------|
 * | fmt             | Data format of the values to be stored in DEST                                               | DataFormat  | See supported formats above                                | True     |
 *
 * | Argument        | Description                                                                                  | Type        | Valid Range                                                | Required |
 * |-----------------|----------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------|----------|
 * | tile_id         | Index of the tile within DEST to write                                                       | uint32_t    | Must be less than the size of the DST register buffer      | True     |
 * | src_buffer      | Source buffer containing TILE_HEIGHT * TILE_WIDTH elements of *fmt*                          | const void* | Non-null                                                   | True     |
 * | enable_swizzle  | When true, write with hardware swizzling enabled (matches the FPU's view of DEST)            | bool        | true or false                                              | False    |
 */
// clang-format on
template <DataFormat fmt>
ALWI void dbg_write_dest_tile(uint32_t tile_id, const void* src_buffer, bool enable_swizzle = true) {
    static_assert(dbg_dest_fmt_supported(fmt), "dbg_write_dest_tile: unsupported DataFormat");
    MATH((dbg_copy_dest_tile<DbgDestTileOp::Write, MathThreadId>(fmt, tile_id, src_buffer, enable_swizzle)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
