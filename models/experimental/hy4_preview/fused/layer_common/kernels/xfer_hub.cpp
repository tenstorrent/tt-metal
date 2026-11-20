// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub of the transport-only program (fused/layer_common/transport.py Transport.xfer_program): the layer's two
// cross-chip steps with no compute between them. On the root chip it first plays the part of the iHC site: its
// source buffer goes to the sender's staging and the broadcast op is made ready. On every chip it then waits
// (bounded) for x to arrive in its x buffer and hands its "partial output" (a buffer the host filled) to the
// reduce: what the hub of a real layer program does around its compute.
//
// Stamps (HY4L_TIMING): 0 start, 1 x arrived, 2 partial handed on.
//
// Runtime args: is_root, source address, source bytes, sender NoC x | y << 16, staging address, ready word address,
//               ready bit of the broadcast op, the partial output's address, then hy4l::feed_reduce's arguments.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "layer_common.hpp"

void kernel_main() {
    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4l::word(hy4l::W_RUNS);
    runs[0] = runs[0] + 1;
    if (get_arg_val<uint32_t>(0)) {
        const uint32_t sender = get_arg_val<uint32_t>(3);
        const uint32_t sx = sender & 0xFFFF;
        const uint32_t sy = sender >> 16;
        noc_async_write(
            get_arg_val<uint32_t>(1), get_noc_addr(sx, sy, get_arg_val<uint32_t>(4)), get_arg_val<uint32_t>(2));
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sx, sy, get_arg_val<uint32_t>(5)), get_arg_val<uint32_t>(6));
        noc_async_atomic_barrier();
    }
    if (hy4l::wait_word(hy4l::W_X, 1, hy4l::ST_X)) {
        hy4l::take_word(hy4l::W_X, 1);
    }
    hy4l::stamp(1);
    hy4l::feed_reduce(8, get_arg_val<uint32_t>(7));
    volatile tt_l1_ptr uint32_t* fed = hy4l::word(hy4l::W_FED);
    fed[0] = fed[0] + 1;
    hy4l::stamp(2);
}
