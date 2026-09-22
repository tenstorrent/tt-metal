// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real producer<->consumer flow control: each round write VALUE_r at the live write pointer.
// Runs more rounds than the ring capacity so slots wrap and get reused.
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
            auto lk = dfb.scoped_write_lock(lock_n);
            // get_ptr() hands out the UNCACHED L1 alias on Quasar DM, so the view must be selected here
            // rather than assumed: NONSNOOP needs the alias, the default variant needs the cacheable
            // address. Normalize in whichever direction is required, tolerating the alias being absent --
            // it is temporary DFB API behaviour (see dataflow_buffer.h get_write_ptr). Blindly adding
            // MEM_L1_UNCACHED_BASE would double-apply it and land at 0x800000+, i.e. in RISC local memory.
            const uint32_t lock_ptr = static_cast<uint32_t>(lk.get_ptr().get_address());
            const bool uncached = lock_ptr >= MEM_L1_UNCACHED_BASE;
#if defined(DFB_CACHE_NONSNOOP_PRODUCER)
            const uint32_t entry_addr = uncached ? lock_ptr : lock_ptr + MEM_L1_UNCACHED_BASE;
#else
            const uint32_t entry_addr = uncached ? lock_ptr - MEM_L1_UNCACHED_BASE : lock_ptr;
#endif
            volatile uint32_t* entry = (volatile uint32_t*)(uintptr_t)entry_addr;
            entry[0] = new_val + r;  // round value VALUE_r
        }
        dfb.push_back(1);
    }
}
