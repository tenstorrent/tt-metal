// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1) of the grouped prefill experts. Per group n: this core's two h tiles (intermediate columns b0 and b1)
// go to slots b0 and b1 of cb_h_all on every core of the team (itself included), then one increment of each team
// core's semaphore; once this core's semaphore counts the team's 4 cores for this group, cb_h_all is complete and
// pushed to compute. Then group n + 1's x: this core's quarter of its x tiles (cb_xq, tilized by compute) goes to slot
// quarter of cb_x on every team core, one increment of each one's x semaphore, and once this core's x semaphore counts
// the team's 4 quarters cb_x is pushed to compute. Every team core has made its h of group n by then, so it is done
// reading its cb_x of group n. Then this core's down output of group n, 32 untilized rows of its 32 adjacent tile
// columns (2 KB each): row r goes to row row_dest[r] (k * T + t) of the k-major row-major output [K * T, H]; padding
// rows (row_dest 0xFFFFFFFF) are skipped. These writes overlap compute's gate | up of group n + 1.
//
// cb_h_all holds two groups (8 tiles each) and every team core walks the same groups, so a sender's group n lands
// in slot n % 2: a core that is still reading slot n % 2 for group n - 2 cannot be overwritten, because the sender
// only has group n's h after its own down of group n - 1, which needed this core's h of group n - 1, which this
// core makes after its down of group n - 2.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_plan = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h_own = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_all = get_compile_time_arg_val(2);
    constexpr uint32_t cb_y = get_compile_time_arg_val(3);
    constexpr uint32_t team_size = get_compile_time_arg_val(4);  // 4
    constexpr uint32_t ikt = get_compile_time_arg_val(5);        // 8 h tiles per group
    constexpr uint32_t nt = get_compile_time_arg_val(6);         // hidden / 32 = 128
    constexpr uint32_t cols = get_compile_time_arg_val(7);       // output columns per bank = 16
    constexpr uint32_t banks = get_compile_time_arg_val(8);      // 8
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_xq = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x = get_compile_time_arg_val(11);
    constexpr uint32_t x_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t quarter_tiles = nt / team_size;
    constexpr uint32_t quarter_bytes = quarter_tiles * 2048;
    constexpr uint32_t cb_dst = get_compile_time_arg_val(13);  // 128 B: one group's row destinations
    constexpr uint32_t seg_bytes = 2 * cols * 64;  // a row of this core's 32 adjacent output columns, bf16: 2 KB
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t bank0 = get_arg_val<uint32_t>(0);
    const uint32_t bank1 = get_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_arg_val<uint32_t>(2);
    uint32_t peer_x[team_size], peer_y[team_size];
    for (uint32_t t = 0; t < team_size; t++) {
        peer_x[t] = get_arg_val<uint32_t>(3 + 2 * t);
        peer_y[t] = get_arg_val<uint32_t>(4 + 2 * t);
    }
    const uint32_t quarter = get_arg_val<uint32_t>(3 + 2 * team_size);
    const uint32_t dest_addr = get_arg_val<uint32_t>(4 + 2 * team_size);
    const uint32_t dest_bytes = get_arg_val<uint32_t>(5 + 2 * team_size);  // the row destinations' page

    cb_wait_front(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_plan));
    const uint32_t jobs = plan[0];

    const uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint32_t h_all_base = get_write_ptr(cb_h_all);
    const uint32_t x_sem_addr = get_semaphore(x_sem_id);
    volatile tt_l1_ptr uint32_t* x_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_sem_addr);
    const uint32_t x_slot = get_write_ptr(cb_x) + quarter * quarter_bytes;
    const InterleavedAddrGen<true> y_gen = {.bank_base_address = y_addr, .page_size = nt * 64};  // a row of H
    const InterleavedAddrGen<true> dest_gen = {.bank_base_address = dest_addr, .page_size = dest_bytes};
    const uint32_t dst_l1 = get_write_ptr(cb_dst);
    volatile tt_l1_ptr uint32_t* row_dest = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);

    uint32_t total = 0;
    for (uint32_t j = 0; j < jobs; j++) {
        total += plan[3 + 3 * j];
    }

    // this core's quarter of group n's x tiles to slot quarter of cb_x on every team core
    auto x_exchange = [&](uint32_t n) {
        cb_reserve_back(cb_x, nt);
        cb_wait_front(cb_xq, quarter_tiles);
        const uint32_t xq = get_read_ptr(cb_xq);
        for (uint32_t t = 0; t < team_size; t++) {
            noc_async_write(xq, get_noc_addr(peer_x[t], peer_y[t], x_slot), quarter_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < team_size; t++) {
            noc_semaphore_inc(get_noc_addr(peer_x[t], peer_y[t], x_sem_addr), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(cb_xq, quarter_tiles);
        noc_semaphore_wait_min(x_sem, team_size * (n + 1));
        cb_push_back(cb_x, nt);
    };

    if (total > 0) {
        x_exchange(0);
    }
    uint32_t n = 0;  // groups done by this team
    for (uint32_t j = 0; j < jobs; j++) {
        const uint32_t first = plan[2 + 3 * j];
        const uint32_t groups = plan[3 + 3 * j];
        for (uint32_t i = 0; i < groups; i++, n++) {
            const uint32_t g = first + i;
            // h exchange
            cb_reserve_back(cb_h_all, ikt);
            cb_wait_front(cb_h_own, 2);
            const uint32_t src = get_read_ptr(cb_h_own);
            const uint32_t slot = h_all_base + (n & 1) * ikt * tile_bytes;
            for (uint32_t t = 0; t < team_size; t++) {
                noc_async_write(src, get_noc_addr(peer_x[t], peer_y[t], slot + bank0 * tile_bytes), tile_bytes);
                noc_async_write(
                    src + tile_bytes, get_noc_addr(peer_x[t], peer_y[t], slot + bank1 * tile_bytes), tile_bytes);
            }
            noc_async_write_barrier();
            for (uint32_t t = 0; t < team_size; t++) {
                noc_semaphore_inc(get_noc_addr(peer_x[t], peer_y[t], sem_addr), 1);
            }
            noc_async_atomic_barrier();
            cb_pop_front(cb_h_own, 2);
            noc_semaphore_wait_min(sem, team_size * (n + 1));
            cb_push_back(cb_h_all, ikt);

            // The next group's x now: every team core has made its h of group n, so it is done with its cb_x.
            if (n + 1 < total) {
                x_exchange(n + 1);
            }

            // this core's down output rows of group g: 32 rows of its 32 adjacent columns (2 KB each)
            noc_async_read(dest_gen.get_noc_addr(0, g * 128), dst_l1, 128);
            noc_async_read_barrier();
            cb_wait_front(cb_y, 2 * cols);
            const uint32_t src_y = get_read_ptr(cb_y);
            const uint32_t col_bytes = quarter * seg_bytes;
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t d = row_dest[r];
#ifdef ABL_NO_Y
                if (false) {
#else
                if (d != 0xFFFFFFFF) {
#endif
                    noc_async_write(src_y + r * seg_bytes, y_gen.get_noc_addr(d, col_bytes), seg_bytes);
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_y, 2 * cols);
        }
    }
    noc_async_write_barrier();
}
