// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mirrors the terminal shape of a persistent service kernel: issue a non-posted
// multicast atomic credit, then return. `drain` selects whether the kernel drains
// the atomic (noc_async_atomic_barrier) before returning.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const std::uint32_t sem_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t x_start = get_arg_val<std::uint32_t>(1);
    const std::uint32_t y_start = get_arg_val<std::uint32_t>(2);
    const std::uint32_t x_end = get_arg_val<std::uint32_t>(3);
    const std::uint32_t y_end = get_arg_val<std::uint32_t>(4);
    const std::uint32_t num_dests = get_arg_val<std::uint32_t>(5);
    const std::uint32_t drain = get_arg_val<std::uint32_t>(6);

    const std::uint64_t mcast_addr = get_noc_multicast_addr(x_start, y_start, x_end, y_end, sem_addr);
    noc_semaphore_inc_multicast(mcast_addr, 1, num_dests);
    if (drain) {
        noc_async_atomic_barrier();
    }
}
