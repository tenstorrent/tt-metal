// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher tile counter logging.
// Producer posts tiles to all DFBs, consumers either ack or exit early based on thread_idx
// This creates predictable TC mismatches for watcher to detect and log

#include "experimental/dataflow_buffer.h"
#include "api/compile_time_args.h"
#include "api/kernel_thread_globals.h"
#include "risc_common.h"

void kernel_main() {
    // Compile-time args: arg[0]=num_dfbs (same for both), arg[1]=entries per thread (differs)
    constexpr uint32_t num_dfbs = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries_per_thread = get_compile_time_arg_val(1);

#ifdef DFB_PRODUCER
    // Producer: post tiles to all DFBs
    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        experimental::DataflowBuffer dfb(dfb_id);
        for (uint32_t entry = 0; entry < num_entries_per_thread; entry++) {
            dfb.reserve_back(1);
            dfb.push_back(1);
        }
    }
#else  // Consumer
    constexpr uint32_t num_consumers_to_run = get_compile_time_arg_val(2);
    uint32_t thread_idx = get_my_thread_id();

    // Early exit for stalled consumers - their TCs won't be acked -> mismatch.
    // Running consumers block on sync flag, keeping kernel alive for watcher to observe.
    if (thread_idx >= num_consumers_to_run) {
        return;
    }

    // Running consumers: ack all entries from all DFBs
    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        experimental::DataflowBuffer dfb(dfb_id);
        for (uint32_t entry = 0; entry < num_entries_per_thread; entry++) {
            dfb.wait_front(1);
            dfb.pop_front(1);
        }
    }

    // Wait on sync flag for host to capture watcher TC state
    uintptr_t sync_flag_addr = static_cast<uintptr_t>(get_common_arg_val<uint32_t>(0));
    volatile uint32_t* sync_flag = reinterpret_cast<volatile uint32_t*>(sync_flag_addr);
    while (*sync_flag != 1) {
        invalidate_l2_cache_line(sync_flag_addr);
    }
#endif
}
