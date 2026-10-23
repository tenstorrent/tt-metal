// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the fused situ_glu binary SFPU op over a tile pair at a time so a host
// test can compare against the torch reference. gate arrives in c_0, up in c_1;
// the result is packed to c_16.
//   compile_time_args = [num_tiles, dst_out_index]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/situ_glu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    // dst index the result is written to: 0 aliases the gate operand (what an expert kernel does
    // to save a dst slot), 2 keeps it separate. Both are exercised by the host test.
    constexpr uint32_t kDstOut = get_compile_time_arg_val(1);
    constexpr uint32_t kDstGate = 0;
    constexpr uint32_t kDstUp = 1;

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer out_cb(cb_out);

    // Both input CBs must share a data format: init_sfpu configures the unpacker from cb_gate,
    // and copy_tile only updates the L1 address per call, not the THCON format registers.
    compute_kernel_hw_startup(cb_gate, cb_out);
    copy_init(cb_gate);
    // Single SFPU op; nothing else reprograms the tanh constants between tiles.
    situ_glu_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        gate_cb.wait_front(1);
        up_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, kDstGate);
        copy_tile(cb_up, 0, kDstUp);
        situ_glu_tile(kDstGate, kDstUp, kDstOut);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(kDstOut, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        gate_cb.pop_front(1);
        up_cb.pop_front(1);
    }
}
