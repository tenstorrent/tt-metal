// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// situ_glu.cpp with the activation on the pack thread (situ_glu_tile_pack): the math thread only copies the gate
// and up tiles into DST. gate arrives in c_0, up in c_1; the result is packed to c_16.
//   compile_time_args = [num_tiles, dst_out_index]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/situ_glu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t kDstOut = get_compile_time_arg_val(1);
    constexpr uint32_t kDstGate = 0;
    constexpr uint32_t kDstUp = 1;

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_gate, cb_out);
    copy_init(cb_gate);
    situ_glu_tile_init_pack();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        gate_cb.wait_front(1);
        up_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, kDstGate);
        copy_tile(cb_up, 0, kDstUp);
        tile_regs_commit();

        // The SFPU frame starts with a DST offset write, a configuration write: hold it until the math thread is done.
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        situ_glu_tile_pack(kDstGate, kDstUp, kDstOut);
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile(kDstOut, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        gate_cb.pop_front(1);
        up_cb.pop_front(1);
    }
}
