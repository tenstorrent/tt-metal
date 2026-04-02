// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compile_time_args.h"
#include "api/kernel_thread_globals.h"
#include "risc_common.h"

// 1 producer - 16 DFBs
// 4 consumers - 16 DFBs = 4 x 16 = 64 TCs
void kernel_main() {
    constexpr uint32_t num_dfbs = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries_per_dfb = get_compile_time_arg_val(1);
#ifdef DFB_PRODUCER
    // go through each DFB
    // push_back each tile
    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        experimental::DataflowBuffer dfb(dfb_id);
        for (uint32_t entry = 0; entry < num_entries_per_dfb; entry++) {
            dfb.reserve_back(1);
            dfb.push_back(1);
        }
    }
#else  // Consumer
    constexpr uint32_t num_entries_per_dfb = get_compile_time_arg_val(2);
    constexpr uint32_t thread_idx = get_my_thread_id();
    // Early exit for stalled consumers
    if (thread_idx >= num_entries_per_dfb) {
        return;  // This consumer never acks -> TC mismatch
    }
    // go through each DFB
    // push_back each tile
    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        experimental::DataflowBuffer dfb(dfb_id);
        for (uint32_t entry = 0; entry < num_entries_per_dfb; entry++) {
            dfb.wait_front(1);
            dfb.pop_front(1);
        }
    }
#endif
    uintptr_t sync_flag_addr = static_cast<uintptr_t>(get_common_arg_val<uint32_t>(0));
    volatile uint32_t* sync_flag = reinterpret_cast<volatile uint32_t*>(sync_flag_addr);
    while (*sync_flag != 1) {
        // invalidate cache
        invalidate_l2_cache_line(sync_flag_addr);
    }
}
