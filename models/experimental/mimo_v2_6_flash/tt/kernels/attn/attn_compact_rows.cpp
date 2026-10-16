// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): rows 0-3 of this core's tile columns [first, first + tiles) of the decode o-projection output
// [1, 1, 32, hidden] (bf16 tiles, L1 or DRAM interleaved) into the compact [1, 1, 32, hidden / 8] DRAM tensor laid out
// like moe_down_writer.cpp's: rows 4c..4c+3 of compact tile t % compact_tiles, c = t / compact_tiles. The row
// all-gather of it and ReduceExpand then all-reduce only the user rows.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(2);
    constexpr bool in_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t rows_bytes = 4 * 16 * 2;  // rows 0-3 of one face: 128 B
    constexpr uint32_t blocks_per_face = face_bytes / rows_bytes;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);

    // Rows 0-3 of faces 0 and 1 of every tile: one [4, 32] tiny tile each.
    const InterleavedAddrGen<in_dram> in_gen = {.bank_base_address = in_addr, .page_size = tile_bytes};
    cb_reserve_back(cb_rows, tiles);
    const uint32_t l1 = get_write_ptr(cb_rows);
    for (uint32_t i = 0; i < tiles; i++) {
        const uint32_t dst = l1 + i * 2 * rows_bytes;
        noc_async_read(in_gen.get_noc_addr(first + i, 0), dst, rows_bytes);
        noc_async_read(in_gen.get_noc_addr(first + i, face_bytes), dst + rows_bytes, rows_bytes);
    }
    noc_async_read_barrier();

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < tiles; i++) {
        const uint32_t t = first + i;
        const uint32_t blk = t / compact_tiles;
        const uint32_t page = t - blk * compact_tiles;
        const uint32_t off = (blk / blocks_per_face) * 2 * face_bytes + (blk % blocks_per_face) * rows_bytes;
        const uint32_t src = l1 + i * 2 * rows_bytes;
        noc_async_write(src, out_gen.get_noc_addr(page, off), rows_bytes);
        noc_async_write(src + rows_bytes, out_gen.get_noc_addr(page, off + face_bytes), rows_bytes);
    }
    noc_async_write_barrier();
}
