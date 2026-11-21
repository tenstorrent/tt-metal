// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE gate|up + SwiGLU, reader (BRISC) (dflash/grouped_moe.py GroupedMoE.gate_up). Units (gmoe_common.hpp):
// (expert e, local column tile j of 23); per unit the expert's slot rows of X_g (all K tiles, once per expert while
// they fit 4 slots, else per 4-slot group per unit), the gate and up column tiles of global column g0(e) + j
// interleaved [k][gate, up] (the relaid weights, gmoe_relayout.cpp: GMOE_SPLIT consecutive pages in consecutive DRAM
// banks, alternating NOCs), and the two column tiles of the 32-row gate / up bias; GMOE_DEPTH units buffered, their
// reads in flight together (dynamic NOC mode).
//
// common runtime args: [plan table, xg, gate_up (relaid), gate_bias, up_bias, (GMOE_XG_ARG:) X_g's grid NOC x per
// column, NOC y per row]
// compile-time args: [cb_scr, cb_meta_c, cb_meta_w, cb_x, cb_w, cb_b, NC, GX, KT, NT, NU, WC, CC,
//                     accessors (table, xg, gate_up, gate_bias, up_bias)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

void kernel_main() {
    const uint32_t table_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t xg_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gu_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t gb_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t ub_addr = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta_c = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t cb_b = get_compile_time_arg_val(5);
    constexpr uint32_t NC = get_compile_time_arg_val(6);
    constexpr uint32_t GX = get_compile_time_arg_val(7);
    constexpr uint32_t KT = get_compile_time_arg_val(8);   // K tiles (hidden / 32)
    constexpr uint32_t NT = get_compile_time_arg_val(9);   // gate / up width in tiles (E * 22.5)
    constexpr uint32_t NU = get_compile_time_arg_val(10);  // column tiles per expert (23)
    constexpr uint32_t WC = get_compile_time_arg_val(11);
    constexpr uint32_t CC = get_compile_time_arg_val(12);
    constexpr auto table_args = TensorAccessorArgs<13>();
    constexpr auto xg_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<xg_args.next_compile_time_args_offset()>();
    constexpr auto gb_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto ub_args = TensorAccessorArgs<gb_args.next_compile_time_args_offset()>();
    constexpr uint32_t x_tile = 1088;       // BFP8_b
    constexpr uint32_t w_tile = 576;        // BFP4_b
    constexpr uint32_t b_tile = 2048;       // BF16
    constexpr uint32_t GROUP = 4;           // slots per in0 load
    constexpr uint32_t DEPTH = GMOE_DEPTH;  // units in flight
    constexpr uint8_t NOC_B = 1;            // the odd pages of each unit (dynamic NOC mode)

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const auto s_table = TensorAccessor(table_args, table_addr, 8 * gmoe::E);
    constexpr uint32_t SPLIT = GMOE_SPLIT;            // pages per unit (gmoe_relayout.cpp GMOE_RELAYOUT_SPLIT)
    constexpr uint32_t sp = 2 * KT * w_tile / SPLIT;  // bytes per page
    static_assert(2 * KT % SPLIT == 0 && sp % 64 == 0);
    const auto s_gu = TensorAccessor(gu_args, gu_addr, sp);
    const auto s_gb = TensorAccessor(gb_args, gb_addr, b_tile);
    const auto s_ub = TensorAccessor(ub_args, ub_addr, b_tile);

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

    // one in0 block = GROUP slots' K tiles (fixed size: the CB holds whole blocks, never one straddling its end)
    constexpr uint32_t X_TRID = DEPTH + 1;
    auto load_x = [&](uint32_t s0, uint32_t n) {
        cb_reserve_back(cb_x, GROUP * KT);
        const uint32_t l1 = get_write_ptr(cb_x);
        // one slot's KT tiles per transaction-id wait: a transaction id counts at most 255 outstanding requests (8
        // bits); 4 slots' 360 reads under one id wrapped it, and the wait returned before every tile had landed
        noc_async_read_set_trid(X_TRID, 0);
        noc_async_read_set_trid(X_TRID, 1);
        // X_g block sharded over a GMOE_XG_SX x GMOE_XG_SY core grid (the gather matmul's own output): slot s's row
        // tile is row s % RS of the shards of core row s / RS, its KT tiles KT / SX contiguous tiles in each core of
        // that row (common args GMOE_XG_ARG..: the grid's NOC x per column, then NOC y per row): SX reads per slot,
        // half on each NOC
        constexpr uint32_t SX = GMOE_XG_SX, CS = KT / SX, RS = GMOE_XG_RS;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t s = s0 + i;
            const uint32_t ny = get_common_arg_val<uint32_t>(GMOE_XG_ARG + SX + s / RS);
            for (uint32_t j = 0; j < SX; ++j) {
                const uint32_t nx = get_common_arg_val<uint32_t>(GMOE_XG_ARG + j);
                const uint8_t nn = j % 2;
                noc_async_read(
                    get_noc_addr(nx, ny, xg_addr + (s % RS) * CS * x_tile, nn),
                    l1 + (i * KT + j * CS) * x_tile,
                    CS * x_tile,
                    nn);
            }
        }
        noc_async_read_barrier_with_trid(X_TRID, 0);
        noc_async_read_barrier_with_trid(X_TRID, 1);
        noc_async_read_set_trid(0, 0);
        noc_async_read_set_trid(0, 1);
        cb_push_back(cb_x, GROUP * KT);
    };
    // weights + bias of up to DEPTH units in flight (transaction id 1 + slot on both NOCs; slot = unit count % DEPTH:
    // the CBs hold exactly DEPTH units, fresh at slot 0); a unit's in0 rows are loaded when its weights are pushed
    // (loading them at issue could block on cb_x while the compute waits for those weights)
    constexpr uint32_t PW = 2 * KT;  // weight tiles per unit
    const uint32_t wbase = get_write_ptr(cb_w), bbase = get_write_ptr(cb_b);
    uint32_t ring[DEPTH];
    uint32_t issued = 0, pushed = 0, loaded = gmoe::E;
    auto push_one = [&]() {
        const uint32_t slot = pushed % DEPTH, u = ring[slot];
        const uint32_t e = u / NU, m = p.tiles[e];
        // a new expert's rows (one block) while its weights land: every earlier unit is pushed, so the compute frees
        // the old block without waiting on this one (a multi-block expert loads after the push: its second block waits
        // for the compute to finish the first, which needs these weights)
        if (m <= GROUP && e != loaded) {
            load_x(p.base[e], m);
            loaded = e;
        }
        noc_async_read_barrier_with_trid(1 + slot, 0);
        noc_async_read_barrier_with_trid(1 + slot, NOC_B);
        cb_push_back(cb_w, PW);
        cb_push_back(cb_b, 2);
        ++pushed;
        if (m > GROUP) {
            loaded = gmoe::E;
            for (uint32_t g = 0; g < m; g += GROUP) {
                load_x(p.base[e] + g, m - g < GROUP ? m - g : GROUP);
            }
        }
    };
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t e = u / NU, j = u % NU;
        if (p.tiles[e] == 0) {
            continue;
        }
#ifdef GMOE_GU_GMAJOR
        if (p.tiles[e] > GROUP) {
            // a big expert (512 / 1024-row passes): the pipeline drained, then per group of GROUP slots its rows of X_g
            // once and this core's units of it (their weights read once per group, not the rows once per unit)
            while (pushed < issued) {
                push_one();
            }
            const uint32_t m = p.tiles[e];
            const uint32_t ue1 = u1 < (e + 1) * NU ? u1 : (e + 1) * NU;
            for (uint32_t g = 0; g < m; g += GROUP) {
                load_x(p.base[e] + g, m - g < GROUP ? m - g : GROUP);
                for (uint32_t uu = u; uu < ue1; ++uu) {
                    const uint32_t slot = issued % DEPTH;
                    cb_reserve_back(cb_w, PW);
                    cb_reserve_back(cb_b, 2);
                    const uint32_t wl1 = wbase + slot * PW * w_tile, bl1 = bbase + slot * 2 * b_tile;
                    const uint32_t n = gmoe::g0(e) + uu % NU;
                    for (uint32_t q = 0; q < SPLIT; ++q) {
                        noc_async_read(s_gu.get_noc_addr(uu * SPLIT + q, 0, 0), wl1 + q * sp, sp, 0);
                    }
                    noc_async_read(s_gb.get_noc_addr(n, 0, 0), bl1, b_tile, 0);
                    noc_async_read(s_ub.get_noc_addr(n, 0, 0), bl1 + b_tile, b_tile, 0);
                    noc_async_read_barrier(0);
                    cb_push_back(cb_w, PW);
                    cb_push_back(cb_b, 2);
                    ++issued;
                    ++pushed;
                }
            }
            loaded = gmoe::E;
            u = ue1 - 1;
            continue;
        }
#endif
        if (issued - pushed == DEPTH) {
            push_one();
        }
        const uint32_t slot = issued % DEPTH, inflight = issued - pushed + 1;
        cb_reserve_back(cb_w, inflight * PW);
        cb_reserve_back(cb_b, inflight * 2);
        const uint32_t wl1 = wbase + slot * PW * w_tile, bl1 = bbase + slot * 2 * b_tile;
        const uint32_t n = gmoe::g0(e) + j;  // global column tile
        noc_async_read_set_trid(1 + slot, 0);
        noc_async_read_set_trid(1 + slot, NOC_B);
        // the unit's [k][gate, up] tiles (SPLIT consecutive pages of the relaid weights: consecutive DRAM banks),
        // alternating NOCs
        for (uint32_t q = 0; q < SPLIT; ++q) {
            const uint8_t nn = q % 2 ? NOC_B : 0;
            noc_async_read(s_gu.get_noc_addr(u * SPLIT + q, 0, nn), wl1 + q * sp, sp, nn);
        }
        noc_async_read(s_gb.get_noc_addr(n, 0, 0), bl1, b_tile, 0);
        noc_async_read(s_ub.get_noc_addr(n, 0, NOC_B), bl1 + b_tile, b_tile, NOC_B);
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
