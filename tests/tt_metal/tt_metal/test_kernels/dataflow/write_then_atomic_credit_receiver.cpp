// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver half of the write->atomic-credit ordering probe.
//
// The sender writes a payload into this core's L1, drains only with
// noc_async_writes_flushed() (request SENT, not committed), then credits us with a
// remote atomic increment. Per WormholeB0/NoC/Ordering.md the recipient NIU's
// same-VC guarantees enumerate write->read, atomic->atomic and write->write, but
// NOT write->atomic -- so the credit may be applied before the payload commits.
//
// Every word of the payload carries the iteration tag. We check the LAST words,
// which commit last: a stale tag there means we observed the credit before the data.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sender_x = get_arg_val<uint32_t>(0);
    const uint32_t sender_y = get_arg_val<uint32_t>(1);
    const uint32_t payload_addr = get_arg_val<uint32_t>(2);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(3);
    const uint32_t data_sem_addr = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t ack_sem_addr = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t iters = get_arg_val<uint32_t>(6);
    const uint32_t result_addr = get_arg_val<uint32_t>(7);

    volatile tt_l1_ptr uint32_t* data_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_sem_addr);
    volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    const uint32_t words = payload_bytes / sizeof(uint32_t);
    const uint64_t ack_sem_noc_addr = get_noc_addr(sender_x, sender_y, ack_sem_addr);

    uint32_t stale_publishes = 0;  // iterations whose tail was stale at credit time
    uint32_t stale_words = 0;      // total stale words seen across those iterations
    uint32_t first_bad_iter = 0;

    // Check the final 8 words (two 16-byte L1 write streams x 4 words) -- the tail of
    // the packet, which is the last thing to commit.
    const uint32_t check_from = (words > 8) ? (words - 8) : 0;

    for (uint32_t i = 1; i <= iters; ++i) {
        noc_semaphore_wait_min(data_sem, i);

        invalidate_l1_cache();
        uint32_t bad = 0;
        for (uint32_t w = check_from; w < words; ++w) {
            if (payload[w] != i) {
                bad++;
            }
        }
        if (bad) {
            stale_publishes++;
            stale_words += bad;
            if (first_bad_iter == 0) {
                first_bad_iter = i;
            }
        }

        // Release the sender for the next iteration.
        noc_semaphore_inc(ack_sem_noc_addr, 1);
    }

    result[0] = stale_publishes;
    result[1] = stale_words;
    result[2] = first_bad_iter;
    result[3] = iters;

    noc_async_atomic_barrier();
}
