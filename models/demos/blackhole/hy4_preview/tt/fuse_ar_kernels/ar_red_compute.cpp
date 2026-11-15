// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// All-reduce reducer core, TRISC: sums the NS slots (compact 8x32 tiles) of the receive CB into Q output tiles.
// MODE 0: FPU dest-reuse adds (bf16 dest, truncating); MODE 1: fp32 DEST, SFPU fp32 adds (needs fp32_dest_acc_en), one
// rounding at the pack.

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
    constexpr uint32_t MODE = get_compile_time_arg_val(4);
    constexpr uint32_t CB_OWN =
        get_compile_time_arg_val(5);  // own chunk (Q tiles); CB_RECV holds the NS - 1 remote slots
    constexpr uint32_t NR = NS - 1;

    compute_kernel_hw_startup(CB_RECV, CB_RECV, CB_OUT);
    cb_wait_front(CB_OWN, Q);
    cb_wait_front(CB_RECV, NR * Q);
    tile_regs_acquire();
    copy_init(CB_OWN);
    if constexpr (MODE == 0) {
        for (uint32_t i = 0; i < Q; ++i) {
            copy_tile(CB_OWN, i, i);
        }
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_RECV);
        for (uint32_t s = 0; s < NR; ++s) {
            for (uint32_t i = 0; i < Q; ++i) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_RECV, s * Q + i, i);
            }
        }
    } else {
        // accumulator tiles 0..Q-1, one scratch tile Q (fp32 dest: 8 tiles); the own chunk first, then the remote slots
        // in chip order
        add_binary_tile_init();
        for (uint32_t i = 0; i < Q; ++i) {
            copy_tile(CB_OWN, i, i);
        }
        copy_init(CB_RECV);
        for (uint32_t s = 0; s < NR; ++s) {
            for (uint32_t i = 0; i < Q; ++i) {
                copy_tile(CB_RECV, s * Q + i, Q);
                add_binary_tile(i, Q, i);
            }
        }
    }
    tile_regs_commit();
    cb_reserve_back(CB_OUT, Q);
    tile_regs_wait();
    for (uint32_t i = 0; i < Q; ++i) {
        pack_tile(i, CB_OUT);
    }
    tile_regs_release();
    cb_push_back(CB_OUT, Q);
    cb_pop_front(CB_OWN, Q);
    cb_pop_front(CB_RECV, NR * Q);
}
