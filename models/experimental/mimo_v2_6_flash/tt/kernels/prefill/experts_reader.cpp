// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0) of the grouped prefill experts (GroupedExperts in tt/moe_prefill.py). Reads this core's team plan
// (a 4 KB page: [jobs, (expert, first group, groups) per job]) into cb_plan for the other two RISCs, then per job
// this core's slice of the expert's BFP4 weights, kept in L1 for all of the job's groups: from DRAM banks b0 and
// b1, the gate|up rows of w_gate_up (bank b holds gate column b and up column b of every expert, 256 contiguous
// tiles [kt][gate, up]) and its 32 adjacent down output columns 32 * quarter + .. (8 passes of 4, one tile read
// each). A job's first group is read before its weights (compute tilizes it before the previous job's last down, the
// step that frees the previous job's weights). Per group:
// the routing-weight tile, built here from the group's 32 bf16 weights (row-major, one per packed row): row r holds
// weight r in every column; and this core's quarter of the group's x rows: for each packed row its token's columns
// [quarter * kt / 4 * 32, ...) from the row-major x, row r at r * quarter_bytes (compute tilizes them; the writer
// shares the tiles with the team).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_plan = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wgu = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);        // hidden / 32 = 128
    constexpr uint32_t gu_tiles = get_compile_time_arg_val(7);  // gate|up tiles per bank per expert = 2 * kt
    constexpr uint32_t wd_tiles = get_compile_time_arg_val(8);  // down tiles per bank per expert = 8 * 16
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t plan_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cb_wrow = get_compile_time_arg_val(11);        // 64 B: one group's row weights
    constexpr uint32_t weights_bytes = get_compile_time_arg_val(12);  // the row weights' page: rows_max bf16
    constexpr uint32_t cb_xrm = get_compile_time_arg_val(13);         // 32 rows of this core's quarter of x
    constexpr uint32_t team_size = get_compile_time_arg_val(14);
    constexpr uint32_t cb_tok = get_compile_time_arg_val(15);  // 128 B: one group's tokens
    constexpr uint32_t row_bytes = kt * 64;                    // a bf16 row of x
    constexpr uint32_t quarter_bytes = row_bytes / team_size;
    constexpr uint32_t ikt = 8;     // intermediate tiles of a chip's expert slice
    constexpr uint32_t passes = 8;  // down passes of 4 output tiles: 32 output columns per core

    const uint32_t plan_addr = get_arg_val<uint32_t>(0);
    const uint32_t team = get_arg_val<uint32_t>(1);
    const uint32_t bank0 = get_arg_val<uint32_t>(2);
    const uint32_t bank1 = get_arg_val<uint32_t>(3);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(4);
    const uint32_t wd_addr = get_arg_val<uint32_t>(5);
    const uint32_t x_addr = get_arg_val<uint32_t>(6);
    const uint32_t rw_addr = get_arg_val<uint32_t>(7);
    const uint32_t tok_addr = get_arg_val<uint32_t>(8);
    const uint32_t quarter = get_arg_val<uint32_t>(9);        // this core's index in its team
    const uint32_t tokens_bytes = get_arg_val<uint32_t>(10);  // the row tokens' page

    const InterleavedAddrGen<true> plan_gen = {.bank_base_address = plan_addr, .page_size = plan_bytes};
    cb_reserve_back(cb_plan, 1);
    const uint32_t plan_l1 = get_write_ptr(cb_plan);
    noc_async_read(plan_gen.get_noc_addr(team, 0), plan_l1, plan_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1);
    const uint32_t jobs = plan[0];

    const InterleavedAddrGen<true> x_gen = {.bank_base_address = x_addr, .page_size = row_bytes};
    const InterleavedAddrGen<true> tok_gen = {.bank_base_address = tok_addr, .page_size = tokens_bytes};
    const uint32_t tok_l1 = get_write_ptr(cb_tok);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    const InterleavedAddrGen<true> rw_gen = {.bank_base_address = rw_addr, .page_size = weights_bytes};
    const uint32_t wrow_l1 = get_write_ptr(cb_wrow);
    volatile tt_l1_ptr uint16_t* wrow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wrow_l1);
    constexpr uint32_t gu_bytes = gu_tiles * w_tile_bytes;

    // One group's routing-weight tile (cb_rw) and this core's quarter of its x rows (cb_xrm).
    auto read_group = [&](uint32_t g) {
        cb_reserve_back(cb_rw, 1);
        noc_async_read(rw_gen.get_noc_addr(0, g * 64), wrow_l1, 64);
        noc_async_read_barrier();
        // bf16 32 x 32 tile: row r, columns 0-15 in face (r / 16) * 2 (row r % 16), columns 16-31 in the next face
        volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rw));
        for (uint32_t r = 0; r < 32; r++) {
            const uint32_t v = wrow[r];
            const uint32_t word = v | (v << 16);
            volatile tt_l1_ptr uint32_t* left = tile + (r >> 4) * 256 + (r & 15) * 8;
            for (uint32_t w = 0; w < 8; w++) {
                left[w] = word;
                left[128 + w] = word;
            }
        }
        cb_push_back(cb_rw, 1);
        noc_async_read(tok_gen.get_noc_addr(0, g * 128), tok_l1, 128);
        noc_async_read_barrier();
        cb_reserve_back(cb_xrm, kt / team_size);
        uint32_t dst = get_write_ptr(cb_xrm);
#ifndef ABL_NO_X
        for (uint32_t r = 0; r < 32; r++) {
            noc_async_read(x_gen.get_noc_addr(tok[r], quarter * quarter_bytes), dst, quarter_bytes);
            dst += quarter_bytes;
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_xrm, kt / team_size);
    };

    for (uint32_t j = 0; j < jobs; j++) {
        const uint32_t expert = plan[1 + 3 * j];
        const uint32_t first = plan[2 + 3 * j];
        const uint32_t groups = plan[3 + 3 * j];

        // The job's first group before its weights: compute tilizes it before the previous job's last down, which
        // frees the previous job's weights.
        read_group(first);
        cb_reserve_back(cb_wgu, 2 * gu_tiles);
        cb_reserve_back(cb_wd, 2 * wd_tiles);
        const uint32_t gu_l1 = get_write_ptr(cb_wgu);
        const uint32_t wd_l1 = get_write_ptr(cb_wd);
#ifdef ABL_NO_W
        if (j == 0) {
#endif
            noc_async_read(get_noc_addr_from_bank_id<true>(bank0, wgu_addr + expert * gu_bytes), gu_l1, gu_bytes);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank1, wgu_addr + expert * gu_bytes), gu_l1 + gu_bytes, gu_bytes);
            // down: this core's 32 adjacent output columns 32 quarter + 4 pj + c (c < 4) in pass pj: column n sits in
            // bank n % 8 at bank page expert * 128 + kr * 16 + n / 8; landing as [pj][kr][c]
            uint32_t wd_dst = wd_l1;
            for (uint32_t pj = 0; pj < passes; pj++) {
                for (uint32_t kr = 0; kr < ikt; kr++) {
                    for (uint32_t c = 0; c < 4; c++) {
                        const uint32_t n = 32 * quarter + 4 * pj + c;
                        const uint32_t page = expert * wd_tiles + kr * 16 + n / 8;
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(n % 8, wd_addr + page * w_tile_bytes),
                            wd_dst,
                            w_tile_bytes);
                        wd_dst += w_tile_bytes;
                    }
                }
            }
#ifdef ABL_NO_W
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_wgu, 2 * gu_tiles);
        cb_push_back(cb_wd, 2 * wd_tiles);

        for (uint32_t i = 1; i < groups; i++) {
            read_group(first + i);
        }
    }
}
