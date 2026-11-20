// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Test only (tests/test_handover.py): the party that hands ops over to the collective's sender in its "hand-over
// words" form (fused/collective/kernels/collective_sender.cpp, HY4_COLLECTIVE_HANDOVERS), on the sender core's
// second data-movement processor, as the X4 coordinator of the attention half will. For every hand-over in the
// order of its argument list it
//   - fills the op's staging with a pattern (word i = op << 24 | hand-over number << 16 | i),
//   - writes the op's hand-over words (unless the entry says "no words": the op's own bytes, slot and staging),
//   - adds the op's ready bit,
//   - waits (bounded) for the op's bit in the served word and takes it off.
// A wait that runs out leaves 0xE1 << 24 | op << 8 | hand-over in the status word and the kernel ends.
//
// Runtime args: ready word address, served word address, hand-over table address, status word address, number of
// hand-overs, then per hand-over: op, staging address (of this hand-over), staging bytes, 1 (write the words) | 0,
// payload bytes, landing offset, staging offset (what the words say; the pattern is written at the address given).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef HANDOVER_TEST_WAIT
#define HANDOVER_TEST_WAIT 30000000u
#endif

void kernel_main() {
    const uint32_t ready = get_arg_val<uint32_t>(0);
    const uint32_t served_addr = get_arg_val<uint32_t>(1);
    const uint32_t table = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    const uint32_t n = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* served = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(served_addr);
    constexpr uint32_t HEAD = 5, PER = 7;
    uint32_t number[16] = {0};

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t op = get_arg_val<uint32_t>(HEAD + PER * i);
        volatile tt_l1_ptr uint32_t* staging =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(HEAD + PER * i + 1));
        const uint32_t words = get_arg_val<uint32_t>(HEAD + PER * i + 2) / 4;
        for (uint32_t w = 0; w < words; ++w) {
            staging[w] = (op << 24) | (number[op] << 16) | w;
        }
        if (get_arg_val<uint32_t>(HEAD + PER * i + 3) != 0) {
            volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table + 16 * op);
            words[1] = get_arg_val<uint32_t>(HEAD + PER * i + 5);
            words[2] = get_arg_val<uint32_t>(HEAD + PER * i + 6);
            words[0] = (1u << 31) | get_arg_val<uint32_t>(HEAD + PER * i + 4);
        }
        noc_semaphore_inc(get_noc_addr(ready), 1u << (2 * op));
        noc_async_atomic_barrier();
        bool seen = false;
        for (uint32_t polls = 0; polls < HANDOVER_TEST_WAIT; ++polls) {
            invalidate_l1_cache();
            if ((served[0] >> op) & 1u) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            status[0] = (0xE1u << 24) | (op << 8) | number[op];
            return;
        }
        noc_semaphore_inc(get_noc_addr(served_addr), 0u - (1u << op));
        noc_async_atomic_barrier();
        ++number[op];
    }
}
