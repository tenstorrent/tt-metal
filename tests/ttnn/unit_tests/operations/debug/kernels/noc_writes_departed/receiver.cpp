// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks every slot the sender writes: the expected value is sent_tag(pass, i). overwrite_tag(pass, i) means
// the NoC read the sender's source word after the sender overwrote it, i.e. after the flush had returned.

#include "common.hpp"

using namespace noc_writes_departed;

void kernel_main() {
    constexpr uint32_t RING = get_compile_time_arg_val(0);
    constexpr uint32_t PASSES = get_compile_time_arg_val(1);

    const Layout l(get_arg_val<uint32_t>(0), RING);
    const uint32_t sender_x = get_arg_val<uint32_t>(1);
    const uint32_t sender_y = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* res = l1_word(l.results);
    for (uint32_t k = 0; k < RESULT_WORDS; ++k) {
        res[k] = 0;
    }
    uint32_t num_samples = 0;
    for (uint32_t pass = 0; pass < PASSES; ++pass) {
        for (uint32_t i = 0; i < RING; ++i) {
            volatile tt_l1_ptr uint32_t* slot = l1_word(l.base + i * SLOT_BYTES);
            uint32_t value;
            do {
                invalidate_l1_cache();
                value = *slot;
            } while (value == 0);
            *slot = 0;
            if (value == sent_tag(pass, i)) {
                res[RES_OK]++;
                continue;
            }
            const bool previous = (i > 0 && value == overwrite_tag(pass, i - 1)) ||
                                  (i == 0 && pass > 0 && value == overwrite_tag(pass - 1, RING - 1));
            if (value == overwrite_tag(pass, i)) {
                res[RES_STALE]++;
            } else if (previous) {
                res[RES_PREVIOUS]++;
            } else {
                res[RES_OTHER]++;
            }
            if (num_samples < NUM_SAMPLES) {
                res[RES_SAMPLES + 2 * num_samples] = (pass << 16) | i;
                res[RES_SAMPLES + 2 * num_samples + 1] = value;
                num_samples++;
            }
        }
        res[RES_PASSES] = pass + 1;
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, l.ack), 1);
        noc_async_atomic_barrier();
    }
}
