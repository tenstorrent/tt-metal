// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Same-core consumer for the SenderPipe INCLUDE_SRC regression. The sender data-movement
// kernel publishes cb_in immediately after send() returns; this compute kernel deliberately
// has no semaphore receive/wait beyond the CB publication and therefore requires send() to
// have completed the sender's loopback destination write before returning.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    CircularBuffer input(cb_in);
    CircularBuffer output(cb_out);
    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        input.wait_front(1);
        output.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        input.pop_front(1);
        output.push_back(1);
    }
}
