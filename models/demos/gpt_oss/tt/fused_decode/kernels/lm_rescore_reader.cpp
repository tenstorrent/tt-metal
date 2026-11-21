// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage DFlash LM head, stage 2 with column-tile de-duplication (NCRISC) (dflash/lm_rescore.py).
// Cores: a K x rows rectangle, entry e = r K + j (row r, candidate j) on (x0 + j, y0 + r).
// A. The row cores (j = 0): merge row r's stage-1 lists (one 8 K byte page per (stage-1 core, row)) into the row's K
//    best columns (ties: lower index), write them to entries r K .. r K + K - 1 of the hub's (entry 0's core) table
//    (cb_tab, 4 B per entry) and increment the hub's program semaphore 0.
// B. The hub waits for all rows, then multicasts the table and sets program semaphore 1 on every core.
//    The hub also reads x (all rows, [rows, 32] row-tile layout) once and multicasts it before the table (every core
//    reading it from the boundary helpers made them a hot spot).
// C. Every core: its entry's column tile n_e = c_e / 32; if an earlier entry has the same tile, nothing to do
//    (mailbox 0 to the writer / compute); else read tile n_e of the BF8 stream (kt tiles; round-robin layout of
//    dflash/lm_head.py: tile n on bank n % RS_RR, slot n / RS_RR) and run the [rows, 32] x tile matmul (the writer
//    fills in every entry of n_e).
//
// runtime args: [lists1_addr, x_addr, w_addr, r, j, rect noc x0, y0, x1, y1, hub noc x, y]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t lists_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t r = get_arg_val<uint32_t>(3);
    const uint32_t j = get_arg_val<uint32_t>(4);
    const uint32_t mx0 = get_arg_val<uint32_t>(5), my0 = get_arg_val<uint32_t>(6);
    const uint32_t mx1 = get_arg_val<uint32_t>(7), my1 = get_arg_val<uint32_t>(8);
    const uint32_t hx = get_arg_val<uint32_t>(9), hy = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t n1 = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t kx = get_compile_time_arg_val(7);
    constexpr uint32_t col_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t nbias = get_compile_time_arg_val(9);
    constexpr auto l_args = TensorAccessorArgs<10>();
    constexpr auto x_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr uint32_t row_tile = 64 * rows;  // one [rows, 32] BF16 tile
    constexpr uint32_t n_ent = rows * K;
    const auto s_l = TensorAccessor(l_args, lists_addr, K * 8);
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t e = r * K + j;
    const uint32_t tab = get_write_ptr(cb_tab);
    volatile tt_l1_ptr uint32_t* sem0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));

    // x (all rows, [rows, 32] row-tile layout, 2 KB pages) + the bias tile: read once by the hub, multicast below
    cb_reserve_back(cb_x, kx + 1);
    const uint32_t xl = get_write_ptr(cb_x);
    constexpr uint32_t x_bytes = kx * row_tile;
    if (e == 0) {
        for (uint32_t p = 0; p * 2048 < x_bytes; ++p) {
            const uint32_t nb = x_bytes - p * 2048 < 2048 ? x_bytes - p * 2048 : 2048;
            noc_async_read(s_x.get_noc_addr(p), xl + p * 2048, nb);
        }
    }
    volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(xl + kx * row_tile);
    for (uint32_t i = 0; i < 32 * rows; ++i) {
        ones[i] = (i < 16 * rows && i % 16 < nbias) ? 0x3F80 : 0;  // the bias rows (as the stream writers)
    }

    if (j == 0) {  // A
        cb_reserve_back(cb_scr, 1);
        const uint32_t scr = get_write_ptr(cb_scr);
        for (uint32_t i = 0; i < n1; ++i) {
            noc_async_read(s_l.get_noc_addr(i * rows + r), scr + i * K * 8, K * 8);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr int32_t* q = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
        int32_t bk[K];
        uint32_t bi[K];
        for (uint32_t k = 0; k < K; ++k) {
            bk[k] = INT32_MIN;
            bi[k] = 0xFFFFFFFFu;
        }
        for (uint32_t t = 0; t < n1 * K; ++t) {
            const int32_t key = q[2 * t];
            const uint32_t idx = static_cast<uint32_t>(q[2 * t + 1]);
            if (key > bk[K - 1] || (key == bk[K - 1] && idx < bi[K - 1])) {
                uint32_t p = K - 1;
                while (p > 0 && (key > bk[p - 1] || (key == bk[p - 1] && idx < bi[p - 1]))) {
                    bk[p] = bk[p - 1];
                    bi[p] = bi[p - 1];
                    --p;
                }
                bk[p] = key;
                bi[p] = idx;
            }
        }
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t k = 0; k < K; ++k) {
            o[k] = bi[k];
        }
        noc_async_write(scr, get_noc_addr(hx, hy, tab + 4 * r * K), 4 * K);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hx, hy, get_semaphore(0)), 1);
        noc_async_atomic_barrier();
    }
    if (e == 0) {  // B: the hub (x and the table to every core)
        noc_async_read_barrier();
        noc_async_write_multicast(xl, get_noc_multicast_addr(mx0, my0, mx1, my1, xl), x_bytes, n_ent - 1);
        noc_semaphore_wait(sem0, rows);
        noc_async_write_multicast_loopback_src(tab, get_noc_multicast_addr(mx0, my0, mx1, my1, tab), 4 * n_ent, n_ent);
        noc_async_write_barrier();
        sem1[0] = 1;
        noc_semaphore_set_multicast_loopback_src(
            get_semaphore(1), get_noc_multicast_addr(mx0, my0, mx1, my1, get_semaphore(1)), n_ent);
        noc_async_write_barrier();
    }
    noc_semaphore_wait(sem1, 1);  // C
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* t_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    const uint32_t n = t_[e] / 32;
    bool owner = true;
    for (uint32_t i = 0; i < e; ++i) {
        owner &= (t_[i] / 32) != n;
    }
    if (owner) {
        // (x arrived with the table) the column tile (one read: reading only the needed 16-column half's faces, 3
        // small reads per K tile, measured slower)
        cb_reserve_back(cb_w, kx + 1);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(n % RS_RR, w_addr + (n / RS_RR) * col_bytes),
            get_write_ptr(cb_w),
            col_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_x, kx + 1);
        cb_push_back(cb_w, kx + 1);
    }
}
