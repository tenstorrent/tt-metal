// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): decode SDPA output [1, B, heads, D] (bf16 tiles, interleaved; user b's head h is row h % 32 of its
// tile row h / 32) -> concatenated heads [1, 1, B, heads * vd] (bf16 tiles, L1 interleaved): row b of output tile
// column h * v_tiles + dt is row h of user b's tile column dt, for dt < v_tiles (the V padding columns are dropped).
// One unit is the 16-row half (faces 0-1 or 2-3) of one output tile; this core does units [first, first + units).
// With a mask (owner mask [1, 1, B, 1], bf16 0 / 1): the rows of users whose mask is 0 are zeros (the mul by it).
// The SDPA output may hold fewer users than the 32 output rows (a compact decode batch): rows >= in_users are zeros.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t units = get_compile_time_arg_val(1);
    constexpr uint32_t user_pages = get_compile_time_arg_val(2);  // tiles per user in the SDPA output
    constexpr uint32_t d_tiles = get_compile_time_arg_val(3);     // tile columns of the SDPA output (padded D / 32)
    constexpr uint32_t v_tiles = get_compile_time_arg_val(4);     // kept tile columns per head (vd / 32)
    constexpr bool in_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool has_mask = get_compile_time_arg_val(6) == 1;
    constexpr bool mask_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t in_users = get_compile_time_arg_val(8);  // users in the SDPA output (<= 32)
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t half_bytes = 2 * face_bytes;  // rows 0-15 or 16-31 of a tile: two faces
    constexpr uint32_t row_bytes = 32;               // one face row: 16 bf16

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);

    // Scratch: units 16-row halves (1 KB each), then the mask tile.
    cb_reserve_back(cb_scratch, units + 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t mask_l1 = scratch + units * half_bytes;
    if constexpr (has_mask) {
        const InterleavedAddrGen<mask_dram> mask_gen = {.bank_base_address = mask_addr, .page_size = tile_bytes};
        noc_async_read(mask_gen.get_noc_addr(0, 0), mask_l1, tile_bytes);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint16_t* mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1);

    const InterleavedAddrGen<in_dram> in_gen = {.bank_base_address = in_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < units; i++) {
        const uint32_t unit = first + i;
        const uint32_t tc = unit >> 1;
        const uint32_t half = unit & 1;
        const uint32_t h = tc / v_tiles;
        const uint32_t dt = tc - h * v_tiles;
        const uint32_t page0 = (h >> 5) * d_tiles + dt;
        const uint32_t src_off = ((h >> 4) & 1) * half_bytes + (h & 15) * row_bytes;
        const uint32_t dst = scratch + i * half_bytes;
        for (uint32_t r = 0; r < 16; r++) {
            const uint32_t b = half * 16 + r;
            const uint32_t d0 = dst + r * row_bytes;
            const uint32_t d1 = d0 + face_bytes;
            bool zero = b >= in_users;
            if constexpr (has_mask) {
                // element (b, 0) of the mask tile: face (b / 16) * 2, row b % 16
                zero = zero || (mask[(b >> 4) * (half_bytes / 2) + (b & 15) * 16] & 0x7fffu) == 0;
            }
            if (zero) {
                volatile tt_l1_ptr uint32_t* z0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d0);
                volatile tt_l1_ptr uint32_t* z1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d1);
                for (uint32_t w = 0; w < row_bytes / 4; w++) {
                    z0[w] = 0;
                    z1[w] = 0;
                }
                continue;
            }
            const uint32_t page = b * user_pages + page0;
            noc_async_read(in_gen.get_noc_addr(page, src_off), d0, row_bytes);
            noc_async_read(in_gen.get_noc_addr(page, src_off + face_bytes), d1, row_bytes);
        }
    }
    noc_async_read_barrier();

    const InterleavedAddrGen<false> out_gen = {.bank_base_address = out_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < units; i++) {
        const uint32_t unit = first + i;
        noc_async_write(scratch + i * half_bytes, out_gen.get_noc_addr(unit >> 1, (unit & 1) * half_bytes), half_bytes);
    }
    noc_async_write_barrier();
}
