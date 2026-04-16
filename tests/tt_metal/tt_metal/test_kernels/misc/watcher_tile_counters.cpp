// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher tile counter logging (DM -> NEO)
// DM producer posts tiles, 2 NEO TRISC0 consume and 2 NEO TRISC0 exit early
// Creates predictable TC mismatches for watcher to detect and log

#include "experimental/dataflow_buffer.h"
#include "api/compile_time_args.h"
#include "api/kernel_thread_globals.h"
#include "risc_common.h"
#include "api/debug/dprint.h"
#if defined(COMPILE_FOR_DM)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(UCK_CHLKC_UNPACK)
#include "internal/tt-2xx/quasar/tensix_neo_reg.h"
#endif

constexpr uint32_t dfb_id = get_compile_time_arg_val(0);
constexpr uint32_t num_entries = get_compile_time_arg_val(1);

void kernel_main() {
#if defined(COMPILE_FOR_DM)
    // DM Producer: post tiles to DFB for all 4 NEO consumers
    experimental::DataflowBuffer dfb(dfb_id);
    for (uint32_t entry = 0; entry < num_entries; entry++) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }

#elif defined(UCK_CHLKC_UNPACK)
    // NEO TRISC0 Consumer: consume tiles from DFB
    constexpr uint32_t num_consumers_to_run = get_compile_time_arg_val(2);
    constexpr uint32_t sync_flag_addr = get_compile_time_arg_val(3);
    uint32_t thread_idx = get_my_thread_id();

    // Early exit for stalled consumers - their TCs won't be acked -> mismatch
    if (thread_idx >= num_consumers_to_run) {
        return;
    }

    // Running consumers: consume all entries
    experimental::DataflowBuffer dfb(dfb_id);
    for (uint32_t entry = 0; entry < num_entries; entry++) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }

    // Wait on sync flag for host to capture watcher TC state
    volatile uint32_t* sync_flag = reinterpret_cast<volatile uint32_t*>(sync_flag_addr);
    while (*sync_flag != 1) {
    }
#endif
}
