// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Appends IMAGE_ID << 16 | its second runtime arg to a log in L1: word 0 counts the runs, and runs past
// the third runtime arg (capacity) are counted but not written. Each image compiles with its own
// IMAGE_ID, so a stale binary, runtime arg or install shows up in the log.
//
// With LATE_ACK_MCAST it then multicasts a non-posted write and returns without waiting for its acks,
// which the kernel-end contract allows in dedicated mode (writes sent, not acked).

#include <cstdint>
#include <utility>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t log_addr = get_arg_val<uint32_t>(0);
    auto* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_addr);
#if defined(LATE_ACK_PROBE)
    // A write and its barrier first: the barrier hangs if this kernel's counters were synced while the
    // previous image's acks were still arriving.
    noc_async_write(
        log_addr,
        get_noc_addr(get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7) + get_arg_val<uint32_t>(8)),
        64);
    noc_async_write_barrier();
#endif
    const uint32_t n = log[0];
    if (n < get_arg_val<uint32_t>(2)) {
        log[1 + n] = (IMAGE_ID << 16) | get_arg_val<uint32_t>(1);
    }
    log[0] = n + 1;

#if defined(LATE_ACK_MCAST)
    uint32_t start_x = get_arg_val<uint32_t>(3);
    uint32_t start_y = get_arg_val<uint32_t>(4);
    uint32_t end_x = get_arg_val<uint32_t>(5);
    uint32_t end_y = get_arg_val<uint32_t>(6);
    if (noc_index == 1) {
        std::swap(start_x, end_x);
        std::swap(start_y, end_y);
    }
    const uint64_t dst = get_noc_multicast_addr(start_x, start_y, end_x, end_y, get_arg_val<uint32_t>(7));
    noc_async_write_multicast(log_addr, dst, get_arg_val<uint32_t>(8), get_arg_val<uint32_t>(9));
#if defined(LATE_ACK_BARRIER)
    noc_async_write_barrier();
#endif
#endif
}
