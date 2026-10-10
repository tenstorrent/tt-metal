// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sends RING * PASSES 4 B NoC writes from one L1 word (CELL). Each iteration:
//   CELL = sent_tag; [optional payload write]; write CELL to slot i of the receiver(s); flush;
//   CELL = overwrite_tag
// If the flush returns before the NoC has read CELL, a receiver can get overwrite_tag.
// This is the pattern of a multicast sender that resets its own semaphore right after flushing.

#include "common.hpp"

using namespace noc_writes_departed;

void kernel_main() {
    constexpr uint32_t RING = get_compile_time_arg_val(0);
    constexpr uint32_t PASSES = get_compile_time_arg_val(1);
    constexpr uint32_t FLUSH_MODE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_RECEIVERS = get_compile_time_arg_val(3);
    constexpr bool MCAST = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t DATA_BYTES = get_compile_time_arg_val(5);

    const Layout l(get_arg_val<uint32_t>(0), RING);
    // Unicast: (start_x, start_y) is the receiver. Multicast: the rectangle, already ordered for this NoC.
    const uint32_t start_x = get_arg_val<uint32_t>(1);
    const uint32_t start_y = get_arg_val<uint32_t>(2);
    const uint32_t end_x = get_arg_val<uint32_t>(3);
    const uint32_t end_y = get_arg_val<uint32_t>(4);

    volatile tt_l1_ptr uint32_t* cell = l1_word(l.cell);
    volatile tt_l1_ptr uint32_t* ack = l1_word(l.ack);

    for (uint32_t pass = 0; pass < PASSES; ++pass) {
        for (uint32_t i = 0; i < RING; ++i) {
            const uint32_t slot = l.base + i * SLOT_BYTES;
            if constexpr (DATA_BYTES > 0) {
                if constexpr (MCAST) {
                    noc_async_write_multicast(
                        l.data,
                        get_noc_multicast_addr(start_x, start_y, end_x, end_y, l.data),
                        DATA_BYTES,
                        NUM_RECEIVERS,
                        /*linked=*/true);
                } else {
                    noc_async_write(l.data, get_noc_addr(start_x, start_y, l.data), DATA_BYTES);
                }
            }
            *cell = sent_tag(pass, i);
            if constexpr (MCAST) {
                noc_semaphore_set_multicast(
                    l.cell, get_noc_multicast_addr(start_x, start_y, end_x, end_y, slot), NUM_RECEIVERS);
            } else {
                noc_semaphore_set_remote(l.cell, get_noc_addr(start_x, start_y, slot));
            }
            if constexpr (FLUSH_MODE == FLUSH_WRITES_FLUSHED) {
                noc_async_writes_flushed();
            } else if constexpr (FLUSH_MODE == FLUSH_WRITES_DEPARTED) {
                noc_async_writes_departed();
            }
            *cell = overwrite_tag(pass, i);
        }
        noc_async_write_barrier();
        do {
            invalidate_l1_cache();
        } while (*ack < NUM_RECEIVERS * (pass + 1));
    }
}
