// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The X4 coordinator of a chip in program P2 at 16 token rows (attn_coord.cpp for two sub-passes of 8 rows with a
// row mask; fused/layer_attn/PLAN.md section 19; attn_rows.hpp), on the sender core's second data-movement
// processor. The sender is built with its hand-over words (collective_sender.cpp HY4_COLLECTIVE_HANDOVERS): every
// column op and the row op are handed over once per sub-pass.
//
// For sub-pass p = 0, 1, with n = the live rows of the sub-pass:
//   column stage  for the head groups g in order: wait (bounded) until byte p of C_GROUP[g] has reached n (the
//                 attention cores count their final partial of row 8 p + b there) and take n off; if the group has a
//                 destination: make sure the op's previous hand-over and the one that last used the staging slot
//                 have been served (the served word; each seen bit is taken off), pull the n live rows' partials
//                 into the slot in row order (attention core (b, g) keeps the one of row 8 p + b at
//                 partial + p x bytes), write the op's hand-over words (n partials; landing slot p; staging slot)
//                 and add its ready bit. A sub-pass without a live row is handed over with 0 bytes.
//   row stage     wait until byte p of C_ROW has reached n (the merge cores write their column results into row
//                 staging p themselves), take n off, hand the row op over (n partials; landing slot p; staging p).
// A wait that runs out leaves its code in this core's status word and the kernel carries on, so the program ends.
//
// Runtime args: own group, partial address on an attention core, partial bytes, column staging address, slot bytes,
// ready word address, row op (0xFFFF: none), hand-over table address, then per group 0..7 its op (0xFFFF: none),
// then per S-block 0..7 the NoC x | y << 16 of its attention core of group 0 (group g is at x + g % 4, y + g / 4).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_rows_dm.hpp"

void kernel_main() {
    constexpr uint32_t NONE = 0xFFFF, GROUPS = 8, MAX_OPS = 16;
    const uint32_t own = get_arg_val<uint32_t>(0);
    const uint32_t partial = get_arg_val<uint32_t>(1);
    const uint32_t part_bytes = get_arg_val<uint32_t>(2);
    const uint32_t staging = get_arg_val<uint32_t>(3);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(4);
    const uint32_t ready = get_arg_val<uint32_t>(5);
    const uint32_t row_op = get_arg_val<uint32_t>(6);
    const uint32_t table = get_arg_val<uint32_t>(7);
    constexpr uint32_t OPS = 8, BLOCKS = 16;

    volatile tt_l1_ptr uint32_t* served = hy4a::word(hy4a::A_SERVED);
    served[0] = 0;
    const uint32_t mask = hy4a::row_mask();

    bool open[MAX_OPS] = {false};  // handed over, its served bit not yet seen and taken off
    // The op's last hand-over has been served: its staging may be written and the op may be handed over again.
    auto settle = [&](uint32_t op) {
        if (op == NONE || !open[op]) {
            return;
        }
        uint32_t seen = 0;
        for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
            invalidate_l1_cache();
            seen = served[0];
            if ((seen >> op) & 1u) {
                break;
            }
        }
        if ((seen >> op) & 1u) {
            noc_semaphore_inc(get_noc_addr(hy4a::word_addr(hy4a::A_SERVED)), 0u - (1u << op));
            noc_async_atomic_barrier();
        } else {
            hy4a::fail(hy4a::ST_SERVED, seen);
        }
        open[op] = false;
    };
    auto hand_over = [&](uint32_t op, uint32_t bytes, uint32_t landing_offset, uint32_t staging_offset) {
        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table + 16 * op);
        words[1] = landing_offset;
        words[2] = staging_offset;
        words[0] = (1u << 31) | bytes;
        noc_semaphore_inc(get_noc_addr(ready), 1u << (2 * op));
        noc_async_atomic_barrier();
        open[op] = true;
    };

    uint32_t last[2] = {NONE, NONE};  // the op that last used each column staging slot
    uint32_t sent = 0;
    for (uint32_t p = 0; p < hy4r::SUBPASSES; ++p) {
        const uint32_t n = hy4r::live_count(mask, p);
        for (uint32_t g = 0; g < GROUPS; ++g) {
            if (g == own) {
                continue;
            }
            if (n != 0 && hy4a::wait_field(hy4a::C_GROUP + g, p, n, hy4a::ST_GROUP)) {
                hy4a::take_field(hy4a::C_GROUP + g, p, n);
            }
            const uint32_t op = get_arg_val<uint32_t>(OPS + g);
            if (op == NONE) {
                continue;
            }
            const uint32_t slot = sent & 1u;
            settle(last[slot]);
            settle(op);
            const uint32_t dst = staging + slot * slot_bytes;
            hy4r::for_each_live(mask, p, [&](uint32_t b, uint32_t k) {
                const uint32_t a0 = get_arg_val<uint32_t>(BLOCKS + b);
                noc_async_read(
                    get_noc_addr((a0 & 0xFFFF) + g % 4, (a0 >> 16) + g / 4, partial + p * part_bytes),
                    dst + k * part_bytes,
                    part_bytes);
            });
            noc_async_read_barrier();
            hand_over(op, n * part_bytes, p * slot_bytes, slot * slot_bytes);
            last[slot] = op;
            ++sent;
        }
        if (row_op != NONE) {
            if (n != 0 && hy4a::wait_field(hy4a::C_ROW, p, n, hy4a::ST_ROW)) {
                hy4a::take_field(hy4a::C_ROW, p, n);
            }
            settle(row_op);
            hand_over(row_op, n * part_bytes, p * slot_bytes, p * slot_bytes);
        }
    }
}
