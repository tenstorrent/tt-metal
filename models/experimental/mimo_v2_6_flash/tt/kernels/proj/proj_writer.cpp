// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC: the user rows 0..rows-1 (rows 4, 8 or 16) of every bf16 x tile into cb_x (block_rows tiles per push, the order
// compute consumes), then this core's output tiles: the user rows from the packed results, the other rows zero.
//
// x arrives one of three ways (runtime arg role): 0 reads every tile from the x tensor; 1 (the loader of a bank group)
// does the same and, after each block, raises the x semaphore of each of its peers by block_rows; 2 (a peer) waits
// until its loader's count covers the block and copies it from the loader's cb_x, which sits at the same L1 address.
// One read per tile covers the user rows of faces 0 and 1 (and face 0's other rows between them).
//
// Only the user rows of x carry users and an output row depends only on the same x row, so the unread rows of cb_x
// reach only output rows that are written as zeros.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <bool dram>
FORCE_INLINE uint64_t page_noc_addr(uint32_t base, uint32_t page, uint32_t page_bytes) {
    constexpr uint32_t banks = dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    return get_noc_addr_from_bank_id<dram>(page % banks, base + (page / banks) * page_bytes);
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(2);
    constexpr bool x_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t col_stride = get_compile_time_arg_val(7);
    constexpr uint32_t sem_id = get_compile_time_arg_val(8);
    // > 0: write the user rows of output tile column n into the compact rows layout instead (DRAM tiles: rows
    // rows * c .. of compact tile n % compact_tiles, c = n / compact_tiles), the layout the row reduction moves.
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t rows = get_compile_time_arg_val(10);  // the user rows: 4, 8 or 16
    constexpr uint32_t tile_bytes = 2048;                    // bf16 32x32
    constexpr uint32_t face_bytes = 512;                     // bf16 16x16
    constexpr uint32_t user_bytes = rows * 32;               // the user rows of a face
    constexpr uint32_t blocks_per_face = 16 / rows;
    constexpr uint32_t x_read_bytes = face_bytes + user_bytes;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_col = get_arg_val<uint32_t>(2);
    const uint32_t valid_cols = get_arg_val<uint32_t>(3);
    const uint32_t role = get_arg_val<uint32_t>(4);
    const uint32_t src_x = get_arg_val<uint32_t>(5);
    const uint32_t src_y = get_arg_val<uint32_t>(6);
    const uint32_t num_peers = get_arg_val<uint32_t>(7);

    const uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    cb_reserve_back(cb_x, num_blocks * block_rows);
    uint32_t x_l1 = get_write_ptr(cb_x);
    uint32_t k = 0;
    for (uint32_t b = 0; b < num_blocks; b++) {
        if (role == 2) {
            noc_semaphore_wait_min(sem, k + block_rows);
            for (uint32_t r = 0; r < block_rows; r++, k++) {
                noc_async_read(get_noc_addr(src_x, src_y, x_l1), x_l1, x_read_bytes);
                x_l1 += tile_bytes;
            }
        } else {
            for (uint32_t r = 0; r < block_rows; r++, k++) {
                noc_async_read(page_noc_addr<x_dram>(x_addr, k, tile_bytes), x_l1, x_read_bytes);
                x_l1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, block_rows);
        if (role == 1) {
            for (uint32_t p = 0; p < num_peers; p++) {
                const uint32_t px = get_arg_val<uint32_t>(8 + 2 * p);
                const uint32_t py = get_arg_val<uint32_t>(9 + 2 * p);
                noc_semaphore_inc(get_noc_addr(px, py, sem_addr), block_rows);
            }
        }
    }

    const uint32_t zero_l1 = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t i = 0; i < tile_bytes / 4; i++) {
        zero[i] = 0;
    }

    cb_wait_front(cb_out, cols);
    uint32_t src = get_read_ptr(cb_out);
    if constexpr (compact_tiles > 0) {
        for (uint32_t j = 0; j < valid_cols; j++) {
            const uint32_t n = first_col + j * col_stride;
            const uint32_t c = n / compact_tiles;
            const uint64_t dst = page_noc_addr<true>(out_addr, n % compact_tiles, tile_bytes) +
                                 (c / blocks_per_face) * 2 * face_bytes + (c % blocks_per_face) * user_bytes;
            noc_async_write(src, dst, user_bytes);
            noc_async_write(src + face_bytes, dst + face_bytes, user_bytes);
            src += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, cols);
        if (role == 1) {
            noc_async_atomic_barrier();
        }
        return;
    }
    for (uint32_t j = 0; j < valid_cols; j++) {
        const uint64_t dst = page_noc_addr<false>(out_addr, first_col + j * col_stride, tile_bytes);
        noc_async_write(src, dst, user_bytes);
        if constexpr (face_bytes > user_bytes) {  // 16 user rows fill face 0
            noc_async_write(zero_l1, dst + user_bytes, face_bytes - user_bytes);
        }
        noc_async_write(src + face_bytes, dst + face_bytes, user_bytes);
        noc_async_write(zero_l1, dst + face_bytes + user_bytes, tile_bytes - face_bytes - user_bytes);
        src += tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
    if (role == 1) {
        noc_async_atomic_barrier();
    }
}
