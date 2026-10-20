// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row gather, routing-weight tiles, pass table and output scatter (BRISC, NOC1) of the per-expert dispatch gate|up
// op (custom_ops/expert_stream/op.py, ExpertGateUpDispatch).
//
// Per expert of this core's group (dispatch_common.hpp assignment) and per 32-row pass of the tokens routed to it:
// 1. gathers the pass's token rows of the ROW_MAJOR activation x [T, K] (one 8 KB NOC read per row) into a 32-row
//    row-major block that the compute tilizes (rows past the pass's count keep the previous content: finite values
//    that the zero routing weight annihilates; the first pass fills them with its first row);
// 2. builds D = diag(w[t_i, e]) (BF16, exact) for the pass's rows;
// 3. writes the finished [32, 32] tiles to act [1, P, 32, I] (tile page = pass * I_tiles + bank * ppb + q);
// 4. on bank 0 writes the pass table row [n_rows, e, t_0 .. t_31] (uint16) that the down op dispatches from.
// The compute learns each expert's pass count through the BRISC mailboxes (0 = done).
//
// runtime args: [x_addr, routing_addr (EP: weights), counts_addr (EP: idx), act_addr, table_addr, bank_id, group,
//                (EP) base_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "dispatch_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t act_addr = get_arg_val<uint32_t>(3);
    const uint32_t table_addr = get_arg_val<uint32_t>(4);
    const uint32_t bank_id = get_arg_val<uint32_t>(5);
    const uint32_t group = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_xrm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t cb_d = get_compile_time_arg_val(3);
    constexpr uint32_t cb_t = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t kt = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    constexpr uint32_t T = get_compile_time_arg_val(8);
    constexpr uint32_t groups = get_compile_time_arg_val(9);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t table_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ppb = get_compile_time_arg_val(13);  // column pairs per bank = output tiles per pass
    constexpr uint32_t EP = get_compile_time_arg_val(14);   // 1: routing from the top-k index list + weights + base
    constexpr uint32_t K = get_compile_time_arg_val(15);
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto routing_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();       // EP: weights
    constexpr auto counts_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();  // EP: idx
    constexpr auto act_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t RT = T / 32;
    constexpr uint32_t tile_page = 2048;
    static_assert(RT * 32 == T, "T must be a multiple of 32");

    const auto s_x = TensorAccessor(x_args, x_addr, row_bytes);
    const auto s_r = TensorAccessor(routing_args, routing_addr, tile_page);
    const auto s_c = TensorAccessor(counts_args, counts_addr, tile_page);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    const auto s_tab = TensorAccessor(table_args, table_addr, table_bytes);

    constexpr uint32_t c_tiles = EP ? RT : E_TILES;       // EP: the index tiles
    constexpr uint32_t r_tiles = EP ? RT : RT * E_TILES;  // EP: the weight tiles
    cb_reserve_back(cb_c, c_tiles);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    for (uint32_t i = 0; i < c_tiles; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * tile_page, tile_page);
    }
    cb_reserve_back(cb_r, r_tiles);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    for (uint32_t i = 0; i < r_tiles; ++i) {
        noc_async_read(s_r.get_noc_addr(i), r_l1 + i * tile_page, tile_page);
    }
    uint32_t expert_base = 0;
    uint16_t n_tok[E], passes[E], base[E], mine[E];
    if constexpr (EP) {
        const uint32_t base_addr = get_arg_val<uint32_t>(7);
        const auto s_b = TensorAccessor(base_args, base_addr, 32);
        cb_reserve_back(cb_c, 1);
        const uint32_t b_l1 = c_l1 + c_tiles * tile_page;
        noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
        noc_async_read_barrier();
        expert_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
        count_from_indices<E, T, K>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), expert_base, n_tok);
    } else {
        noc_async_read_barrier();
        read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    }
    pass_bases<E>(n_tok, passes, base);
    const uint32_t n_mine = assign_group<E, 32>(passes, groups, group, mine);
    const volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);

    uint16_t rows[T];
    uint16_t wbits[T];
    bool first_pass = true;
    for (uint32_t i = 0; i < n_mine; ++i) {
        const uint32_t e = mine[i];
        // the tokens routed to e, ascending, and their routing weights
        uint32_t nr = 0;
        if constexpr (EP) {
            nr = rows_from_indices<T, K>(
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), routing, expert_base + e, rows, wbits);
        } else {
            for (uint32_t t = 0; t < T; ++t) {
                const volatile tt_l1_ptr uint16_t* tile = routing + ((t >> 5) * E_TILES + (e >> 5)) * 1024;
                if (tile[tile_elem(t & 31, e & 31)] != 0) {
                    wbits[nr] = tile[tile_elem(t & 31, e & 31)];
                    rows[nr++] = static_cast<uint16_t>(t);
                }
            }
        }
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, passes[e]);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, passes[e]);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, passes[e]);
        for (uint32_t p = 0; p < passes[e]; ++p) {
            const uint32_t i0 = p * 32;
            const uint32_t nr_p = (nr - i0) < 32 ? (nr - i0) : 32;
            const uint32_t k = base[e] + p;
            if (bank_id == 0) {  // the pass table row
                cb_reserve_back(cb_t, 1);
                const uint32_t t_l1 = get_write_ptr(cb_t);
                volatile tt_l1_ptr uint16_t* tab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t_l1);
                tab[0] = static_cast<uint16_t>(nr_p);
                tab[1] = static_cast<uint16_t>(e);
                for (uint32_t j = 0; j < 32; ++j) {
                    tab[2 + j] = j < nr_p ? rows[i0 + j] : 0xFFFF;
                }
                noc_async_write(t_l1, s_tab.get_noc_addr(k), table_bytes);
                noc_async_write_barrier();
            }
            // the pass's rows of x -> the row-major block (row j at j * row_bytes)
            cb_reserve_back(cb_xrm, kt);
            const uint32_t xrm = get_write_ptr(cb_xrm);
            for (uint32_t j = 0; j < nr_p; ++j) {
                noc_async_read(s_x.get_noc_addr(rows[i0 + j]), xrm + j * row_bytes, row_bytes);
            }
            if (first_pass) {
                for (uint32_t j = nr_p; j < 32; ++j) {
                    noc_async_read(s_x.get_noc_addr(rows[i0]), xrm + j * row_bytes, row_bytes);
                }
                first_pass = false;
            }
            // D = diag(w[t_j, e]) for the pass's rows
            cb_reserve_back(cb_d, 1);
            const uint32_t d_l1 = get_write_ptr(cb_d);
            volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_l1);
            for (uint32_t j = 0; j < tile_page / 4; ++j) {
                d32[j] = 0;
            }
            volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d_l1);
            for (uint32_t j = 0; j < nr_p; ++j) {
                d16[tile_elem(j, j)] = wbits[i0 + j];
            }
            noc_async_read_barrier();
            cb_push_back(cb_xrm, kt);
            cb_push_back(cb_d, 1);

            for (uint32_t q = 0; q < ppb; ++q) {
                cb_wait_front(cb_out, 1);
                noc_async_write(get_read_ptr(cb_out), s_act.get_noc_addr(k * i_tiles + bank_id * ppb + q), tile_page);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 0);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, 0);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 0);
}
