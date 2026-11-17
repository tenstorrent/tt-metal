// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumes the bound `target` DFB. Popping entry 0 moves this kernel's read pointer to entry 1 while its write pointer
// stays on entry 0, as this kernel never reserves. The sender multicasts entry 1 through read_ptr_view, and each
// receiver asserts that the pattern landed at its own read pointer.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer target(dfb::target);
    // Raised on the sender by each receiver once its producer has filled its entries.
    Semaphore ready(sem::ready);
    // Raised on the receivers by the sender once its multicast has landed.
    Semaphore valid(sem::valid);
    Noc noc;
    const uint32_t sender_x = get_arg(args::sender_x);
    const uint32_t sender_y = get_arg(args::sender_y);
    constexpr uint32_t num_words = get_arg(args::size_bytes) / sizeof(uint32_t);
    target.wait_front(2);
    target.pop_front(1);
    if (noc.is_local_bank(sender_x, sender_y)) {
        ready.wait(get_arg(args::num_targets));
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(target.get_read_ptr()),
            read_ptr_view(target),
            get_arg(args::size_bytes),
            get_arg(args::num_targets),
            {},
            {
                .noc_x_start = get_arg(args::noc_x_start),
                .noc_y_start = get_arg(args::noc_y_start),
                .noc_x_end = get_arg(args::noc_x_end),
                .noc_y_end = get_arg(args::noc_y_end),
            });
        noc.async_write_barrier();
        valid.set(1);
        valid.set_multicast(
            noc,
            get_arg(args::noc_x_start),
            get_arg(args::noc_y_start),
            get_arg(args::noc_x_end),
            get_arg(args::noc_y_end),
            get_arg(args::num_targets));
        noc.async_write_barrier();
    } else {
        ready.up(noc, sender_x, sender_y, 1);
        // The increment is a NoC atomic; drain it so the kernel doesn't exit with it in flight.
        noc.async_atomic_barrier();
        valid.wait(1);
        auto* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target.get_read_ptr());
        for (uint32_t i = 0; i < num_words; ++i) {
            ASSERT(entry[i] == get_arg(args::pattern) + i);
        }
    }
    target.pop_front(1);
}
