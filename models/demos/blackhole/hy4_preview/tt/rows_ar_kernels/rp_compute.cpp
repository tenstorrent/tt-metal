// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Rows-phase reduce, reducer core (TRISC): sums the own chunk (CB_OWN, Q tiny tiles) and the NR remote slots of CB_RECV
// into Q output tiles, SUB tiles per DEST window (fp32 DEST holds SUB accumulators + 1 scratch tile: SUB + 1 <= 8).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr uint32_t Q = get_compile_time_arg_val(2);
    constexpr uint32_t NS = get_compile_time_arg_val(3);
    constexpr uint32_t CB_OWN = get_compile_time_arg_val(4);
    constexpr uint32_t SUB = get_compile_time_arg_val(5);
    constexpr uint32_t MODE = get_compile_time_arg_val(6);  // debug 1: own chunk only (copy)
    constexpr uint32_t CB_FOLD = get_compile_time_arg_val(7);
    constexpr uint32_t NR = NS - 1;

    compute_kernel_hw_startup(CB_RECV, CB_RECV, CB_OUT);
    cb_wait_front(CB_OWN, Q);
    cb_wait_front(CB_RECV, NR * Q);
#ifdef RP_FOLD
    cb_wait_front(CB_FOLD, Q);
#endif
    cb_reserve_back(CB_OUT, Q);
    for (uint32_t t0 = 0; t0 < Q; t0 += SUB) {
        const uint32_t n = (Q - t0 < SUB) ? (Q - t0) : SUB;
        tile_regs_acquire();
        copy_init(CB_OWN);
        add_binary_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_OWN, t0 + i, i);
        }
#ifdef RP_FOLD
        copy_init(CB_FOLD);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_FOLD, t0 + i, n);
            add_binary_tile(i, n, i);
        }
#endif
        copy_init(CB_RECV);
        for (uint32_t s = 0; s < (MODE ? 0 : NR); ++s) {
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(CB_RECV, s * Q + t0 + i, n);
                add_binary_tile(i, n, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile<true>(i, CB_OUT, t0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(CB_OUT, Q);
    cb_pop_front(CB_OWN, Q);
#ifdef RP_FOLD
    cb_pop_front(CB_FOLD, Q);
#endif
    cb_pop_front(CB_RECV, NR * Q);
}
