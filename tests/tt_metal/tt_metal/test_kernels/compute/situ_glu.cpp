// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the fused situ_glu binary SFPU op over a tile pair at a time so a host
// test can compare against the torch reference. gate arrives in c_0, up in c_1;
// the result is packed to c_16.
//   compile_time_args = [num_tiles]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/situ_glu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer out_cb(cb_out);

    init_sfpu(cb_gate, cb_out);
    // Single SFPU op; nothing else reprograms the tanh constants between tiles.
    situ_glu_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        gate_cb.wait_front(1);
        up_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, 0);  // gate -> dst[0]
        copy_tile(cb_up, 0, 1);    // up   -> dst[1]
        situ_glu_tile(0, 1, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        gate_cb.pop_front(1);
        up_cb.pop_front(1);
    }
}
