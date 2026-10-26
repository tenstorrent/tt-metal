// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation load, routing-weight tiles and output scatter (BRISC, NOC1) for the GLM-5.3-Flash expert gate|up
// stream (custom_ops/expert_stream/op.py).
//
// 1. Scans the routing matrix (same scan as the reader) and tells the compute threads how many experts this
//    group streams (BRISC -> TRISC mailboxes).
// 2. Loads the [M <= 32, K] BF16 activation (kt 32x32 tiles) into L1 once; every expert's gate and up columns are
//    multiplied against it.
// 3. For each of the group's experts e builds the diagonal tile D_e = diag(w[0..M-1, e]) (BF16, exact), which the
//    compute multiplies into the SwiGLU output (D_e @ h scales token row t by its routing weight w[t, e]).
// 4. Writes each [32, 32] output tile (expert e, this bank's intermediate tile columns bank * ppb + q) into the act
//    tensor [1, E, 32, I] (TILE, interleaved: tile page e * I_tiles + bank * ppb + q).
//
// runtime args: [x_addr, routing_addr, act_addr, bank_id, group, (ROUTING_FOLD) base_addr,
//                (FUSED_SIGNAL) n_down, sem_base, sems_per_group, down_xy_0 .. down_xy_{n_down-1}]
//
// E1 variants (decode roofline, 2026-10-26; preprocessor-gated, the default program is the one above):
// * ROUTING_FOLD / SCAN_WORDS: as reader_gate_up.cpp (the routing window of this chip's E experts inside the
//   router's full matrix; the word-wise scan).
// * FUSED_SIGNAL (op.py ExpertStreamFused: gate|up and down in ONE program): after the last act tile of its k-th
//   expert is written and acknowledged, the core adds 1 << (4 * (k % 7)) to semaphore
//   sem_base + group * sems_per_group + k / 7 of every down core (`down_xy`: NOC x << 16 | y): a 4-bit arrival
//   counter per (group, expert-in-group) that the down writer (writer_down.cpp FUSED_WAIT) polls to `banks` before it
//   loads that expert's activation tiles. The incs ride the same NOC as the tile writes, after their write barrier.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "routing_scan.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(3);
    const uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_d = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t E = get_compile_time_arg_val(5);
    constexpr uint32_t M = get_compile_time_arg_val(6);
    constexpr uint32_t groups = get_compile_time_arg_val(7);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t ppb = get_compile_time_arg_val(9);  // column pairs per bank = output tiles per expert
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto routing_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr uint32_t ARG_NEXT = 6;
#else
    constexpr uint32_t ARG_NEXT = 5;
#endif

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t tile_page = 2048;  // BF16 32x32

    const auto s_x = TensorAccessor(x_args, x_addr, tile_page);
    const auto s_r = TensorAccessor(routing_args, routing_addr, tile_page);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);

    // routing matrix -> active experts -> this group's share
#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;  // the worst-case fold window
    cb_reserve_back(cb_r, R_TILES + 1);          // + the base page
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const uint32_t base_addr = get_arg_val<uint32_t>(5);
    const auto s_b = TensorAccessor(base_args, base_addr, 32);
    const uint32_t b_l1 = r_l1 + R_TILES * tile_page;
    noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
    noc_async_read_barrier();
    const uint32_t col_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
    const uint32_t tb = fold_first_tile(col_base);
    const uint32_t col_off = fold_col_off(col_base);
    const uint32_t n_tiles = fold_tiles<E>(col_base);
#else
    constexpr uint32_t R_TILES = E_TILES;
    cb_reserve_back(cb_r, E_TILES);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    constexpr uint32_t tb = 0;
    constexpr uint32_t col_off = 0;
    constexpr uint32_t n_tiles = E_TILES;
#endif
    for (uint32_t i = 0; i < n_tiles; ++i) {
        noc_async_read(s_r.get_noc_addr(tb + i), r_l1 + i * tile_page, tile_page);
    }
    // the activation, resident for the whole op
    cb_reserve_back(cb_x, kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kt; ++k) {
        noc_async_read(s_x.get_noc_addr(k), x_l1 + k * tile_page, tile_page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);
    uint16_t active[E];
#if defined(SCAN_WORDS)
    const uint32_t n = scan_active_words<E, M, R_TILES>(routing, active, col_off, n_tiles);
#elif defined(ROUTING_FOLD)
    const uint32_t n = scan_active_fold<E, M>(routing, active, col_off);
#else
    const uint32_t n = scan_active<E, M>(routing, active);
#endif
    uint32_t n_mine = 0;
    for (uint32_t i = group; i < n; i += groups) {
        ++n_mine;
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, n_mine);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n_mine);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n_mine);

#ifdef FUSED_SIGNAL
    const uint32_t n_down = get_arg_val<uint32_t>(ARG_NEXT);
    const uint32_t sem_base = get_arg_val<uint32_t>(ARG_NEXT + 1);
    const uint32_t sems_per_group = get_arg_val<uint32_t>(ARG_NEXT + 2);
    constexpr uint32_t ARG_XY = ARG_NEXT + 3;
#endif

    uint32_t k_mine = 0;  // index of the expert within this group's list (the fused arrival counter's slot)
    for (uint32_t i = group; i < n; i += groups) {
        const uint32_t e = active[i];
        // D_e: zeros, then the token weights on the diagonal (row t = token t)
        cb_reserve_back(cb_d, 1);
        const uint32_t d_l1 = get_write_ptr(cb_d);
        volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_l1);
        for (uint32_t j = 0; j < tile_page / 4; ++j) {
            d32[j] = 0;
        }
        volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d_l1);
        for (uint32_t t = 0; t < M; ++t) {
            d16[tile_elem(t, t)] = routing[fold_elem(t, e, col_off)];
        }
        cb_push_back(cb_d, 1);

        for (uint32_t q = 0; q < ppb; ++q) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), s_act.get_noc_addr(e * i_tiles + bank_id * ppb + q), tile_page);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
#ifdef FUSED_SIGNAL
        {
            // the expert's tiles of this bank are acknowledged (write barrier above): count the arrival on every
            // down core, in the 4-bit slot of this expert's position in the group's list
            const uint32_t sem_addr = get_semaphore(sem_base + group * sems_per_group + k_mine / 7);
            const uint32_t incr = 1u << (4 * (k_mine % 7));
            for (uint32_t d = 0; d < n_down; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(ARG_XY + d);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, sem_addr), incr);
            }
        }
#endif
        ++k_mine;
    }
    (void)k_mine;
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
