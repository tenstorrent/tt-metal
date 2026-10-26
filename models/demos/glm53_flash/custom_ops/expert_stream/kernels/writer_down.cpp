// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation loader + output writer (BRISC, NOC1) for the GLM-5.3-Flash expert down stream
// (custom_ops/expert_stream/op.py).
//
// 1. Scans the routing matrix and tells the compute threads the active-expert count (mailboxes).
// 2. For every active expert e loads its SwiGLU activation tiles act[e] ([32, I] = seg_tiles 32x32 BF16 tiles of
//    the act tensor [1, E, 32, I]) into the in0 CB: compute multiplies them against the expert's down columns and
//    keeps the running sum over the experts in DST.
// 3. Writes the `cols` accumulated [32, 32] output tiles of this core (output tile columns col0 ..) into
//    out [1, 1, 32, H] (TILE, interleaved: tile page = column). With no active expert the columns are zeroed.
//
// runtime args: [routing_addr, act_addr, out_addr, col0, (ROUTING_FOLD) base_addr,
//                (FUSED_WAIT) gate_up_groups, sem_base, sems_per_group, incs_per_expert]
//
// E1 variants (preprocessor-gated, the default program is the one above):
// * ROUTING_FOLD / SCAN_WORDS: as reader_gate_up.cpp.
// * FUSED_WAIT (op.py ExpertStreamFused: this kernel runs in the same program as the gate|up kernels, on other
//   cores): active expert i (rank i of the scan, the same list the gate|up cores derive) is the (i / G)-th expert of
//   gate|up group i % G, so before loading act[e] the core polls the 4-bit slot (i / G) % 7 of semaphore
//   sem_base + (i % G) * sems_per_group + (i / G) / 7 until it reaches `incs_per_expert` (= the DRAM banks: one
//   gate|up core per bank per group writes one tile row of the expert; writer_gate_up.cpp FUSED_SIGNAL). The
//   semaphores are program-initialised to 0 at every launch (the dispatch writes the initial values) and reset to 0
//   here after the last wait, so a replay sees zeros either way.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "routing_scan.hpp"

void kernel_main() {
    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_r = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(4);
    constexpr uint32_t M = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr auto routing_args = TensorAccessorArgs<7>();
    constexpr auto act_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t ARG_NEXT = 5;
#else
    constexpr uint32_t ARG_NEXT = 4;
#endif

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t tile_page = 2048;

    const auto s_r = TensorAccessor(routing_args, routing_addr, tile_page);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_page);

#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;
    cb_reserve_back(cb_r, R_TILES + 1);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const uint32_t base_addr = get_arg_val<uint32_t>(4);
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
    noc_async_read_barrier();
    uint16_t active[E];
    const volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);
#if defined(SCAN_WORDS)
    const uint32_t n = scan_active_words<E, M, R_TILES>(routing, active, col_off, n_tiles);
#elif defined(ROUTING_FOLD)
    const uint32_t n = scan_active_fold<E, M>(routing, active, col_off);
#else
    const uint32_t n = scan_active<E, M>(routing, active);
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);

#ifdef FUSED_WAIT
    const uint32_t gu_groups = get_arg_val<uint32_t>(ARG_NEXT);
    const uint32_t sem_base = get_arg_val<uint32_t>(ARG_NEXT + 1);
    const uint32_t sems_per_group = get_arg_val<uint32_t>(ARG_NEXT + 2);
    const uint32_t incs_per_expert = get_arg_val<uint32_t>(ARG_NEXT + 3);
#endif

    for (uint32_t i = 0; i < n; ++i) {
#ifdef FUSED_WAIT
        {
            // expert rank i = gate|up group i % G, its (i / G)-th expert: wait for the banks' tiles
            const uint32_t g = i % gu_groups;
            const uint32_t k = i / gu_groups;
            volatile tt_l1_ptr uint32_t* sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_base + g * sems_per_group + k / 7));
            const uint32_t shift = 4 * (k % 7);
            while (((*sem >> shift) & 0xFu) < incs_per_expert) {
                invalidate_l1_cache();
            }
        }
#endif
        cb_reserve_back(cb_a, seg_tiles);
        const uint32_t a_l1 = get_write_ptr(cb_a);
        const uint32_t page0 = static_cast<uint32_t>(active[i]) * seg_tiles;
        for (uint32_t k = 0; k < seg_tiles; ++k) {
            noc_async_read(s_act.get_noc_addr(page0 + k), a_l1 + k * tile_page, tile_page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, seg_tiles);
    }
#ifdef FUSED_WAIT
    // every arrival this launch waited for has landed (each slot was polled to its full count): leave the counters
    // at zero for the next launch (the dispatch re-initialises them too)
    for (uint32_t s = 0; s < gu_groups * sems_per_group; ++s) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_base + s)), 0);
    }
#endif

    if (n == 0) {  // no active expert: this core's output columns are zero
        cb_reserve_back(cb_out, 1);
        const uint32_t z = get_write_ptr(cb_out);
        volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t j = 0; j < tile_page / 4; ++j) {
            z32[j] = 0;
        }
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(z, s_out.get_noc_addr(col0 + c), tile_page);
        }
        noc_async_write_barrier();
        noc_async_full_barrier();  // exit contract (this early return included)
        return;
    }
    cb_wait_front(cb_out, cols);
    const uint32_t o_l1 = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        noc_async_write(o_l1 + c * tile_page, s_out.get_noc_addr(col0 + c), tile_page);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
