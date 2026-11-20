// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The X4 coordinator of a chip in program P2 of the attention half (fused/layer_attn/program.py; PLAN.md section
// 5), on the sender core's second data-movement processor. It feeds the sender's column ops and its row op.
//
// Column stage: the partials of head group g (8 users x [max / sum tile | 16 O tiles], 8,704 B each) go to the chip
// of mesh row g in this chip's column. Every attention core keeps its final partial at one fixed L1 address and
// counts C_GROUP[g] on this core when it is packed. For the groups in order: wait (bounded) for the 8 counts; if
// the group has a destination, pull the 8 partials into a staging slot and add the op's ready bit. There are two
// slots, used in turn: a slot is written again only after the sender has reported the op that last used it as
// served (its served word, collective_sender.cpp), so the staging is 2 groups instead of 7.
// Row stage: the merge cores write their column results into the row staging themselves and count C_ROW; at 8 the
// row op's ready bit is added.
// The served word is zeroed at the start: it holds the bits of the launch before, and nothing of this launch can
// be served before this kernel has made it ready.
// A wait that runs out leaves its code in this core's status word and the kernel carries on, so the program ends.
//
// Runtime args: own group, partial address on an attention core, partial bytes, staging address, slot bytes, ready
// word address, row op (0xFFFF: none), then per group 0..7 its op (0xFFFF: none), then per S-block 0..7 the NoC
// x | y << 16 of its attention core of group 0 (group g is at x + g % 4, y + g / 4).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"

void kernel_main() {
    constexpr uint32_t NONE = 0xFFFF, GROUPS = 8, USERS = 8;
    const uint32_t own = get_arg_val<uint32_t>(0);
    const uint32_t partial = get_arg_val<uint32_t>(1);
    const uint32_t part_bytes = get_arg_val<uint32_t>(2);
    const uint32_t staging = get_arg_val<uint32_t>(3);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(4);
    const uint32_t ready = get_arg_val<uint32_t>(5);
    const uint32_t row_op = get_arg_val<uint32_t>(6);
    constexpr uint32_t OPS = 7, BLOCKS = 15;

    volatile tt_l1_ptr uint32_t* served = hy4a::word(hy4a::A_SERVED);
    served[0] = 0;

    uint32_t last[2] = {NONE, NONE};  // the op that last used each staging slot
    uint32_t sent = 0;
    for (uint32_t g = 0; g < GROUPS; ++g) {
        if (g == own) {
            continue;
        }
        if (hy4a::wait_word(hy4a::C_GROUP + g, USERS, hy4a::ST_GROUP)) {
            hy4a::take_word(hy4a::C_GROUP + g, USERS);
        }
        const uint32_t op = get_arg_val<uint32_t>(OPS + g);
        if (op == NONE) {
            continue;
        }
        const uint32_t slot = sent & 1u;
        if (last[slot] != NONE) {
            uint32_t seen = 0;
            for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
                invalidate_l1_cache();
                seen = served[0];
                if ((seen >> last[slot]) & 1u) {
                    break;
                }
            }
            if (((seen >> last[slot]) & 1u) == 0) {
                hy4a::fail(hy4a::ST_SERVED, seen);
            }
        }
        const uint32_t dst = staging + slot * slot_bytes;
        for (uint32_t u = 0; u < USERS; ++u) {
            const uint32_t a0 = get_arg_val<uint32_t>(BLOCKS + u);
            noc_async_read(
                get_noc_addr((a0 & 0xFFFF) + g % 4, (a0 >> 16) + g / 4, partial), dst + u * part_bytes, part_bytes);
        }
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(ready), 1u << (2 * op));
        noc_async_atomic_barrier();
        last[slot] = op;
        ++sent;
    }
    if (row_op != NONE) {
        if (hy4a::wait_word(hy4a::C_ROW, USERS, hy4a::ST_ROW)) {
            hy4a::take_word(hy4a::C_ROW, USERS);
        }
        noc_semaphore_inc(get_noc_addr(ready), 1u << (2 * row_op));
        noc_async_atomic_barrier();
    }
}
