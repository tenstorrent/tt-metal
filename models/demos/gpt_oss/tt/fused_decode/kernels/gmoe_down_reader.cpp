// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE down, reader (BRISC) (dflash/grouped_moe.py GroupedMoE.down). Units (gmoe_common.hpp): (expert e, output
// column tile c of NU = hidden / 32); per expert its slots' rows of A_g (KD = 23 K tiles each: the expert's local
// intermediate columns, gmoe_gu_*), per unit the down weight tiles of column c at the expert's K tiles g0(e) + k (one
// page of the relaid weights, gmoe_relayout.cpp; the K tile the expert shares with a neighbour has the neighbour's
// 16 rows zeroed, so A_g's neighbour half of that tile adds nothing).
//
// common runtime args: [plan table, ag, down (relaid)]
// compile-time args: [cb_scr, cb_meta_c, cb_meta_w, cb_a, cb_w, NC, GX, KD, NU, WC, CC, MAXS,
//                     accessors (table, ag, down)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

void kernel_main() {
    const uint32_t table_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ag_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t down_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta_c = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_a = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t KD = get_compile_time_arg_val(7);  // K tiles per expert (23)
    constexpr uint32_t NU = get_compile_time_arg_val(8);  // output column tiles (hidden / 32)
    constexpr uint32_t WC = get_compile_time_arg_val(9);
    constexpr uint32_t CC = get_compile_time_arg_val(10);
    constexpr uint32_t MAXS = get_compile_time_arg_val(11);  // slots cb_a holds (8): an expert with more is done in
    // groups of MAXS slots, each group with all of this core's units of the expert (their weights read once per group)
    constexpr auto table_args = TensorAccessorArgs<12>();
    constexpr auto ag_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<ag_args.next_compile_time_args_offset()>();
    constexpr uint32_t a_tile = 1088;       // BFP8_b
    constexpr uint32_t w_tile = 576;        // BFP4_b
    constexpr uint32_t DEPTH = GMOE_DEPTH;  // units in flight
    constexpr uint8_t NOC_B = 1;            // the second half of each weight page (dynamic NOC mode)

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const auto s_table = TensorAccessor(table_args, table_addr, 8 * gmoe::E);
    const auto s_a = TensorAccessor(ag_args, ag_addr, a_tile);
    const auto s_d = TensorAccessor(d_args, down_addr, KD * w_tile);

    cb_reserve_back(cb_scr, 1);
    const uint32_t table_l1 = get_write_ptr(cb_scr);
    noc_async_read(s_table.get_noc_addr(0), table_l1, 8 * gmoe::E);
    noc_async_read_barrier();
    invalidate_l1_cache();
    gmoe::Plan p;
    gmoe::read_table(table_l1, p);
    uint32_t u0, u1;
    gmoe::unit_range(p, NU, WC, CC, c, NC, u0, u1);
    for (uint32_t cb : {cb_meta_c, cb_meta_w}) {
        cb_reserve_back(cb, 1);
        gmoe::put_meta(get_write_ptr(cb), u0, u1, p);
        cb_push_back(cb, 1);
    }

    // weights of up to DEPTH units in flight (transaction id 1 + slot on both NOCs; slot = unit count % DEPTH: cb_w
    // holds exactly DEPTH units, fresh at slot 0); an expert's A_g rows are loaded when its first unit's weights are
    // pushed (a whole MAXS-slot block: cb_a never holds a straddling block)
    constexpr uint32_t A_TRID = DEPTH + 1;
    const uint32_t wbase = get_write_ptr(cb_w);
    uint32_t ring[DEPTH];
    uint32_t issued = 0, pushed = 0, loaded = gmoe::E;
    auto push_one = [&]() {
        const uint32_t slot = pushed % DEPTH, u = ring[slot];
        noc_async_read_barrier_with_trid(1 + slot, 0);
        noc_async_read_barrier_with_trid(1 + slot, NOC_B);
        cb_push_back(cb_w, KD);
        ++pushed;
        const uint32_t e = u / NU, m = p.tiles[e];
        if (e != loaded) {
            cb_reserve_back(cb_a, MAXS * KD);
            const uint32_t l1 = get_write_ptr(cb_a);
            // one slot's KD tiles per transaction-id wait (an id counts at most 255 outstanding requests)
            noc_async_read_set_trid(A_TRID, 0);
            for (uint32_t i = 0; i < m; ++i) {
                // A_g height sharded one slot per core: the slot's KD tiles contiguous, one read
                noc_async_read(s_a.get_noc_addr((p.base[e] + i) * KD, 0, 0), l1 + i * KD * a_tile, KD * a_tile, 0);
                noc_async_read_barrier_with_trid(A_TRID, 0);
            }
            noc_async_read_set_trid(0, 0);
            cb_push_back(cb_a, MAXS * KD);
            loaded = e;
        }
    };
    for (uint32_t u = u0; u < u1; ++u) {
        if (p.tiles[u / NU] == 0) {
            continue;
        }
        if (p.tiles[u / NU] > MAXS) {
            // a big expert (more slots than cb_a holds; 512 / 1024-row passes): the pipeline drained, then per group
            // of MAXS slots its rows of A_g and this core's units of it, one at a time
            while (pushed < issued) {
                push_one();
            }
            const uint32_t e = u / NU, m = p.tiles[e];
            const uint32_t ue1 = u1 < (e + 1) * NU ? u1 : (e + 1) * NU;
            for (uint32_t g = 0; g < m; g += MAXS) {
                const uint32_t n = m - g < MAXS ? m - g : MAXS;
                cb_reserve_back(cb_a, MAXS * KD);
                const uint32_t l1 = get_write_ptr(cb_a);
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_read(
                        s_a.get_noc_addr((p.base[e] + g + i) * KD, 0, 0), l1 + i * KD * a_tile, KD * a_tile, 0);
                    noc_async_read_barrier(0);
                }
                cb_push_back(cb_a, MAXS * KD);
                for (uint32_t uu = u; uu < ue1; ++uu) {
                    cb_reserve_back(cb_w, KD);
                    const uint32_t wl1 = wbase + (issued % DEPTH) * KD * w_tile;
                    noc_async_read(s_d.get_noc_addr(uu, 0, 0), wl1, KD * w_tile, 0);
                    noc_async_read_barrier(0);
                    cb_push_back(cb_w, KD);
                    ++issued;
                    ++pushed;
                }
            }
            loaded = gmoe::E;
            u = ue1 - 1;
            continue;
        }
        if (issued - pushed == DEPTH) {
            push_one();
        }
        const uint32_t slot = issued % DEPTH;
        cb_reserve_back(cb_w, (issued - pushed + 1) * KD);
        const uint32_t wl1 = wbase + slot * KD * w_tile;
        constexpr uint32_t half = (KD * w_tile / 2) & ~63u;
        noc_async_read_set_trid(1 + slot, 0);
        noc_async_read_set_trid(1 + slot, NOC_B);
        // the unit's K tiles (one page of the relaid weights), half on each NOC
        noc_async_read(s_d.get_noc_addr(u, 0, 0), wl1, half, 0);
        noc_async_read(s_d.get_noc_addr(u, half, NOC_B), wl1 + half, KD * w_tile - half, NOC_B);
        noc_async_read_set_trid(0, 0);  // (the trid register outlives the kernel: leave it 0)
        noc_async_read_set_trid(0, NOC_B);
        ring[slot] = u;
        ++issued;
    }
    while (pushed < issued) {
        push_one();
    }
    // the global read barriers too: the transaction-id barriers wait on the per-id outstanding counts, and a response
    // still to be counted in the NOC's received count when the kernel exits leaves the next kernel's global read
    // barrier waiting forever (it hung the board after ~10-60 back-to-back MoE passes)
    noc_async_read_barrier(0);
    noc_async_read_barrier(NOC_B);
}
