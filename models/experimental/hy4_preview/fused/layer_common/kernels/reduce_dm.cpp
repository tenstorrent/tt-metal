// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A reduce core of a fused layer program (fused/layer_common/transport.py): data movement. The chips that head a
// mesh row (the root's column) have `parts` of these; core k owns part k of the [8, hidden] BF16 output.
//
//   stage 1   wait (bounded) for n1 arrivals in this core's stage-1 landing: one from each chip of the row (the
//             other chips' senders over the fabric, this chip's hub over the NoC); hand the landing to the compute
//             kernel (reduce_sum.cpp), which sums the n1 slots.
//   a head that is not the root: the sum goes to this chip's sender core (op k), which sends it into the root's
//             stage-2 landing.
//   the root: the compute kernel has packed the sum into this chip's own stage-2 slot; count it, wait (bounded) for
//             n2 arrivals (one per row), hand the landing to compute, and write its sum, this core's part of the
//             layer output F, to the cores that take it (`targets` equal pieces, in order).
//
// Stamps (HY4L_TIMING): 0 start, 1 stage 1 landed, 2 handed to the sender (the root: stage 2 landed), 3 the root:
// F written.
//
// A wait that runs out leaves its code in this core's status word and the kernel carries on (the landing is summed
// as it is), so the program ends and the host sees the status.
//
// Compile-time args: cb_s1, cb_sum (root: over its own stage-2 slot), cb_s2, cb_f, tiles, n1, n2, is_root
// Runtime args: not the root: sender NoC x | y << 16, staging address of this part, ready word address, ready bit,
//               part bytes;  the root: number of targets, bytes per target, address on a target, then per target its
//               NoC x | y << 16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "layer_common.hpp"

constexpr uint32_t cb_s1 = get_compile_time_arg_val(0);
constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
constexpr uint32_t cb_s2 = get_compile_time_arg_val(2);
constexpr uint32_t cb_f = get_compile_time_arg_val(3);
constexpr uint32_t tiles = get_compile_time_arg_val(4);
constexpr uint32_t n1 = get_compile_time_arg_val(5);
constexpr uint32_t n2 = get_compile_time_arg_val(6);
constexpr bool is_root = get_compile_time_arg_val(7) == 1;

void kernel_main() {
    hy4l::stamp(0);
    if (hy4l::wait_word(hy4l::W_S1, n1, hy4l::ST_S1)) {
        hy4l::take_word(hy4l::W_S1, n1);
    }
    hy4l::stamp(1);
    cb_reserve_back(cb_s1, n1 * tiles);
    cb_push_back(cb_s1, n1 * tiles);

    cb_wait_front(cb_sum, tiles);
    if constexpr (!is_root) {
        const uint32_t sender = get_arg_val<uint32_t>(0);
        const uint32_t staging = get_arg_val<uint32_t>(1);
        const uint32_t ready = get_arg_val<uint32_t>(2);
        const uint32_t bit = get_arg_val<uint32_t>(3);
        const uint32_t bytes = get_arg_val<uint32_t>(4);
        const uint32_t sx = sender & 0xFFFF;
        const uint32_t sy = sender >> 16;
        noc_async_write(get_read_ptr(cb_sum), get_noc_addr(sx, sy, staging), bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sx, sy, ready), bit);
        noc_async_atomic_barrier();
        cb_pop_front(cb_sum, tiles);
        hy4l::stamp(2);
    } else {
        // this chip's row sum is in its own stage-2 slot
        noc_semaphore_inc(get_noc_addr(hy4l::word_addr(hy4l::W_S2)), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_sum, tiles);
        if (hy4l::wait_word(hy4l::W_S2, n2, hy4l::ST_S2)) {
            hy4l::take_word(hy4l::W_S2, n2);
        }
        hy4l::stamp(2);
        cb_reserve_back(cb_s2, n2 * tiles);
        cb_push_back(cb_s2, n2 * tiles);

        cb_wait_front(cb_f, tiles);
        const uint32_t source = get_read_ptr(cb_f);
        const uint32_t targets = get_arg_val<uint32_t>(0);
        const uint32_t bytes = get_arg_val<uint32_t>(1);
        const uint32_t address = get_arg_val<uint32_t>(2);
        for (uint32_t t = 0; t < targets; ++t) {
            const uint32_t core = get_arg_val<uint32_t>(3 + t);
            noc_async_write(source + t * bytes, get_noc_addr(core & 0xFFFF, core >> 16, address), bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_f, tiles);
        hy4l::stamp(3);
    }
}
