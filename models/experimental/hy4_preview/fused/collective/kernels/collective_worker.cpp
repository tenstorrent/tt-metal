// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement kernel of the standalone collectives (fused/collective/collective.py) on the landing core. It is
// the smallest user of collective.hpp: the payload is this core's input tensor (cb_in is laid over it).
//
//   reduce (mode 0): per stage, hand the payload over, wait for the ring, give the landing to the compute kernel
//                    (collective_sum.cpp). A later stage's payload is the stage before's sum, which the compute
//                    kernel leaves in cb_out (laid over the output tensor) and then overwrites. Stages may be over
//                    different sites (the two mesh axes) or use one site several times.
//   gather (mode 1): hand the payload over, wait for the ring, copy the landing (the slots in ring order) into
//                    the output tensor.
//   land (mode 2):   the gather without the copy: the result is the landing itself, which is all a fused program
//                    needs and what the host reads back.
//
// Compile-time args: mode, stages (up to 4), tiles, cb_in, cb_out, then per stage: landing CB of parity 0, of parity 1.
// Runtime args: per stage a hy4_collective::Use; gather and land: then the output tensor's address on this core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "collective.hpp"

constexpr uint32_t mode = get_compile_time_arg_val(0);
constexpr uint32_t stages = get_compile_time_arg_val(1);
constexpr uint32_t tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_in = get_compile_time_arg_val(3);
constexpr uint32_t cb_out = get_compile_time_arg_val(4);
constexpr uint32_t MAX_STAGES = 4;
static_assert(stages >= 1 && stages <= MAX_STAGES, "one to four stages");
constexpr uint32_t landing_cb[MAX_STAGES][2] = {
    {get_compile_time_arg_val(5), get_compile_time_arg_val(6)},
    {get_compile_time_arg_val(7), get_compile_time_arg_val(8)},
    {get_compile_time_arg_val(9), get_compile_time_arg_val(10)},
    {get_compile_time_arg_val(11), get_compile_time_arg_val(12)},
};

void kernel_main() {
    size_t arg = 0;
    hy4_collective::Use use[MAX_STAGES];
    for (uint32_t s = 0; s < stages; ++s) {
        use[s] = hy4_collective::read_use(arg);
    }
    hy4_collective::stamp(use[0].ctrl, hy4_collective::STAMP_START);

    if constexpr (mode != 0) {
        const uint32_t output = get_arg_val<uint32_t>(arg++);
        uint32_t parity = 0;
        hy4_collective::hand_over(use[0], get_write_ptr(cb_in));
        (void)hy4_collective::wait_landed(use[0], parity);
        if constexpr (mode == 1) {
            noc_async_write(
                use[0].landing + parity * use[0].parity_stride,
                get_noc_addr(output),
                use[0].members * use[0].slot_stride);
            noc_async_write_barrier();
        }
        hy4_collective::finish(use[0]);
        return;
    }

    for (uint32_t s = 0; s < stages; ++s) {
        uint32_t parity = 0;
        if (s == 0) {
            hy4_collective::hand_over(use[s], get_write_ptr(cb_in));
        } else {
            cb_wait_front(cb_out, tiles);
            // The stage before's landing has been summed.
            hy4_collective::finish(use[s - 1]);
            hy4_collective::hand_over(use[s], get_read_ptr(cb_out));
            cb_pop_front(cb_out, tiles);
        }
        // On a timeout the pages are pushed all the same, so the compute kernel ends; the status word tells.
        (void)hy4_collective::wait_landed(use[s], parity);
        const uint32_t landing = landing_cb[s][parity];
        cb_reserve_back(landing, use[s].members * tiles);
        cb_push_back(landing, use[s].members * tiles);
    }
    cb_wait_front(cb_out, tiles);
    hy4_collective::finish(use[stages - 1]);
    cb_pop_front(cb_out, tiles);
}
