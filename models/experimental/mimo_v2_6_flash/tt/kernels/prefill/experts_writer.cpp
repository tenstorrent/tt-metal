// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1) of the grouped prefill experts. Per group: this core's two h tiles (intermediate columns b0 and b1)
// go to slots b0 and b1 of cb_h_all on every core of the team (itself included), then one increment of each team
// core's semaphore; once this core's semaphore counts the team's 4 cores for this group, cb_h_all is complete and
// pushed to compute. Then this core's 32 output tiles of the group's down rows go to the packed output [G * 32, H].
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
    constexpr uint32_t out_block = 4;
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t bank0 = get_arg_val<uint32_t>(0);
    const uint32_t bank1 = get_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_arg_val<uint32_t>(2);
    uint32_t peer_x[team_size], peer_y[team_size];
    for (uint32_t t = 0; t < team_size; t++) {
        peer_x[t] = get_arg_val<uint32_t>(3 + 2 * t);
        peer_y[t] = get_arg_val<uint32_t>(4 + 2 * t);
    }

    cb_wait_front(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_plan));
    const uint32_t jobs = plan[0];

    const uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint32_t h_all_base = get_write_ptr(cb_h_all);
    const InterleavedAddrGen<true> y_gen = {.bank_base_address = y_addr, .page_size = tile_bytes};

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

            // this core's down output tiles of group g: bank b0 columns, then bank b1 columns
            for (uint32_t p = 0; p < 2 * cols / out_block; p++) {
                const uint32_t bank = p < cols / out_block ? bank0 : bank1;
                const uint32_t m0 = (p % (cols / out_block)) * out_block;
                cb_wait_front(cb_y, out_block);
                uint32_t src_y = get_read_ptr(cb_y);
                for (uint32_t c = 0; c < out_block; c++) {
                    const uint32_t col = bank + banks * (m0 + c);
                    noc_async_write(src_y, y_gen.get_noc_addr(g * nt + col, 0), tile_bytes);
                    src_y += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_y, out_block);
            }
        }
    }
}
