// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensix test kernel: reaches a DRISC's L1 with a plain on-chip NoC address (no DRAM_L1_NOC_OFFSET) on
// the kernel's NoC, first with a 4-byte non-posted write and then with an atomic increment. Each request
// is polled a bounded number of times and its outcome recorded in Tensix L1, so a request the DRISC NIU
// does not acknowledge reads as NOT_ACKED rather than hanging the kernel.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t ACKED = 1;
constexpr uint32_t NOT_ACKED = 2;
constexpr uint32_t POLL_LIMIT = 1u << 24;

template <typename Flushed>
uint32_t poll(Flushed flushed) {
    for (uint32_t i = 0; i < POLL_LIMIT; i++) {
        if (flushed()) {
            return ACKED;
        }
    }
    return NOT_ACKED;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t results_addr = get_compile_time_arg_val(0);  // Tensix L1: {write outcome, atomic outcome}
    constexpr uint32_t source_addr = get_compile_time_arg_val(1);   // Tensix L1: the word to write
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(3);
    constexpr uint32_t drisc_write_addr = get_compile_time_arg_val(4);   // DRISC private L1 address
    constexpr uint32_t drisc_atomic_addr = get_compile_time_arg_val(5);  // DRISC private L1 address

    auto* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    results[0] = 0;
    results[1] = 0;

    noc_async_write(source_addr, get_noc_addr(drisc_noc_x, drisc_noc_y, drisc_write_addr), sizeof(uint32_t));
    results[0] = poll([] { return ncrisc_noc_nonposted_writes_flushed(noc_index); });
    if (results[0] != ACKED) {
        return;
    }

    noc_semaphore_inc(get_noc_addr(drisc_noc_x, drisc_noc_y, drisc_atomic_addr), 1);
    results[1] = poll([] { return ncrisc_noc_nonposted_atomics_flushed(noc_index); });
}
