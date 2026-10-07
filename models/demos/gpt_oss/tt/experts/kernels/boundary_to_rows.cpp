// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Flat hidden vector -> row 0 of a width-sharded BF16 [32, hidden] tile tensor (tt/decode_boundary.py:
// DecodeBoundary.to_rows), on each shard core (NCRISC). Shard tile t of core c is global tile g = tile0 + t, whose
// row 0 holds hidden values [32 g, 32 g + 32): bytes [64 g, 64 g + 32) of the flat vector go to face 0 row 0 (bytes
// [0, 32)) and the next 32 bytes to face 1 row 0 (bytes [512, 544)); everything else is zero.
//
// runtime args: [x_addr, tile0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t x_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t x_noc_y = get_compile_time_arg_val(3);

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t half_row = 32;

    const uint32_t dst0 = get_write_ptr(cb_out);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint32_t base = dst0 + t * tile_bytes;
        const uint64_t src = get_noc_addr(x_noc_x, x_noc_y, x_addr + (tile0 + t) * 2 * half_row);
        noc_async_read(src, base, half_row);
        noc_async_read(src + half_row, base + face_bytes, half_row);
        noc_async_read(zeros, base + half_row, face_bytes - half_row);
        for (uint32_t z = face_bytes + half_row; z < tile_bytes; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tile_bytes - z < MEM_ZEROS_SIZE ? tile_bytes - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, base + z, n);
        }
    }
    noc_async_read_barrier();
}
