// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real producer<->consumer flow control: each round write VALUE_r at the LIVE get_write_ptr(). Runs
// more rounds than the ring capacity so slots wrap and get reused.
//
// -DDFB_CACHE_NONSNOOP_PRODUCER: write through the uncached alias so the store lands in TL1 WITHOUT
// updating the DM consumer's cache (mimics a non-snooping / Tensix producer); the consumer's
// acquire-invalidate is then load-bearing on a wrapped slot.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"  // MEM_L1_UNCACHED_BASE

void kernel_main() {
    DataflowBuffer dfb(dfb::out);
    if (get_my_thread_id() != 0) {
        return;  // single producer
    }
    const uint32_t lock_n = get_arg(args::lock_n);
    const uint32_t new_val = get_arg(args::new_val);
    const uint32_t num_rounds = get_arg(args::num_rounds);

    for (uint32_t r = 0; r < num_rounds; ++r) {
        dfb.reserve_back(1);
        {
            auto lk = dfb.scoped_lock(lock_n);
#if defined(DFB_CACHE_NONSNOOP_PRODUCER)
            volatile uint32_t* entry = (volatile uint32_t*)(uintptr_t)(dfb.get_write_ptr() + MEM_L1_UNCACHED_BASE);
#else
            volatile uint32_t* entry = (volatile uint32_t*)(uintptr_t)dfb.get_write_ptr();
#endif
            entry[0] = new_val + r;  // round value VALUE_r
        }
        dfb.push_back(1);
    }
}
