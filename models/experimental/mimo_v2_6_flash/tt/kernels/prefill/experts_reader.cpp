// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0) of the grouped prefill experts (GroupedExperts in tt/moe_prefill.py). Reads this core's team plan
// (a 4 KB page: [jobs, (expert, first group, groups) per job]) into cb_plan for the other two RISCs, then per job
// this core's slice of the expert's BFP4 weights, kept in L1 for all of the job's groups: from DRAM banks b0 and
// b1, the gate|up rows of w_gate_up (bank b holds gate column b and up column b of every expert, 256 contiguous
// tiles [kt][gate, up]) and the w_down rows (bank b holds output columns b, b + 8, ...: 128 contiguous tiles
// [kt][16]). Per group: the routing-weight tile (row r = the weight of packed row r) and the group's 128 x tiles
// in blocks of x_block tiles.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_plan = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wgu = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);        // hidden / 32 = 128
    constexpr uint32_t x_block = get_compile_time_arg_val(6);   // x tiles per push
    constexpr uint32_t gu_tiles = get_compile_time_arg_val(7);  // gate|up tiles per bank per expert = 2 * kt
    constexpr uint32_t wd_tiles = get_compile_time_arg_val(8);  // down tiles per bank per expert = 8 * 16
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t plan_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t tile_bytes = 2048;  // bf16 32x32

    const uint32_t plan_addr = get_arg_val<uint32_t>(0);
    const uint32_t team = get_arg_val<uint32_t>(1);
    const uint32_t bank0 = get_arg_val<uint32_t>(2);
    const uint32_t bank1 = get_arg_val<uint32_t>(3);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(4);
    const uint32_t wd_addr = get_arg_val<uint32_t>(5);
    const uint32_t x_addr = get_arg_val<uint32_t>(6);
    const uint32_t rw_addr = get_arg_val<uint32_t>(7);

    const InterleavedAddrGen<true> plan_gen = {.bank_base_address = plan_addr, .page_size = plan_bytes};
    cb_reserve_back(cb_plan, 1);
    const uint32_t plan_l1 = get_write_ptr(cb_plan);
    noc_async_read(plan_gen.get_noc_addr(team, 0), plan_l1, plan_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1);
    const uint32_t jobs = plan[0];

    const InterleavedAddrGen<true> x_gen = {.bank_base_address = x_addr, .page_size = tile_bytes};
    const InterleavedAddrGen<true> rw_gen = {.bank_base_address = rw_addr, .page_size = tile_bytes};
    constexpr uint32_t gu_bytes = gu_tiles * w_tile_bytes;
    constexpr uint32_t wd_bytes = wd_tiles * w_tile_bytes;

    for (uint32_t j = 0; j < jobs; j++) {
        const uint32_t expert = plan[1 + 3 * j];
        const uint32_t first = plan[2 + 3 * j];
        const uint32_t groups = plan[3 + 3 * j];

        cb_reserve_back(cb_wgu, 2 * gu_tiles);
        cb_reserve_back(cb_wd, 2 * wd_tiles);
        const uint32_t gu_l1 = get_write_ptr(cb_wgu);
        const uint32_t wd_l1 = get_write_ptr(cb_wd);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank0, wgu_addr + expert * gu_bytes), gu_l1, gu_bytes);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank1, wgu_addr + expert * gu_bytes), gu_l1 + gu_bytes, gu_bytes);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank0, wd_addr + expert * wd_bytes), wd_l1, wd_bytes);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank1, wd_addr + expert * wd_bytes), wd_l1 + wd_bytes, wd_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_wgu, 2 * gu_tiles);
        cb_push_back(cb_wd, 2 * wd_tiles);

        for (uint32_t i = 0; i < groups; i++) {
            const uint32_t g = first + i;
            cb_reserve_back(cb_rw, 1);
            noc_async_read(rw_gen.get_noc_addr(g, 0), get_write_ptr(cb_rw), tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_rw, 1);
            for (uint32_t k0 = 0; k0 < kt; k0 += x_block) {
                cb_reserve_back(cb_x, x_block);
                uint32_t dst = get_write_ptr(cb_x);
                for (uint32_t k = 0; k < x_block; k++) {
                    noc_async_read(x_gen.get_noc_addr(g * kt + k0 + k, 0), dst, tile_bytes);
                    dst += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_x, x_block);
            }
        }
    }
}
