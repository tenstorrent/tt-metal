// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real producer<->consumer flow control: each round cacheable read at the live read pointer. On a wrapped
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
            auto lk = dfb.scoped_read_lock(lock_n);  // acquire invalidates the held entry
            // The read MUST be cacheable for this test to mean anything -- a stale cached line on a wrapped
            // slot is exactly what the acquire-invalidate has to discard. get_ptr() now hands out the
            // UNCACHED L1 alias on Quasar DM (dataflow_buffer.h), which would bypass the cache entirely and
            // make every round pass without exercising the invalidate, so normalize back to the cacheable
            // address here (tolerating the alias being absent, since it is temporary API behaviour).
            const uint32_t rd_ptr = static_cast<uint32_t>(lk.get_ptr().get_address());
            const uint32_t cached_rd_ptr = rd_ptr >= MEM_L1_UNCACHED_BASE ? rd_ptr - MEM_L1_UNCACHED_BASE : rd_ptr;
            result_uncached[r] = *(const volatile uint32_t*)(uintptr_t)cached_rd_ptr;  // this round's value
        }
        dfb.pop_front(1);
    }
}
