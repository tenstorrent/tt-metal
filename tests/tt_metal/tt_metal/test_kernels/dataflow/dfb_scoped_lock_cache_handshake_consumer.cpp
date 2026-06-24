// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real producer<->consumer flow control: each round cacheable read at the LIVE get_read_ptr(). On a wrapped
// (reused) slot the consumer's own prior-round cached line is stale, so the acquire-invalidate must
// discard it for each round to read fresh. One value per round is recorded to the scratch region (via the
// uncached alias, so it lands in TL1 for the host to read).

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"  // MEM_L1_UNCACHED_BASE

void kernel_main() {
    DataflowBuffer dfb(dfb::in);
    if (get_my_thread_id() != 0) {
        return;  // single consumer
    }
    const uint32_t lock_n = get_arg(args::lock_n);
    const uint32_t num_rounds = get_arg(args::num_rounds);
    const uint32_t result_addr = get_arg(args::result_addr);
    volatile uint32_t* result_uncached = (volatile uint32_t*)(uintptr_t)(result_addr + MEM_L1_UNCACHED_BASE);

    for (uint32_t r = 0; r < num_rounds; ++r) {
        dfb.wait_front(1);
        {
            auto lk = dfb.scoped_lock(lock_n);  // acquire invalidates the held entry
            volatile uint32_t* entry = (volatile uint32_t*)(uintptr_t)dfb.get_read_ptr();
            result_uncached[r] = entry[0];  // cacheable read -> record this round's value (fresh, or stale)
        }
        dfb.pop_front(1);
    }
}
