// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher tile counter logging (DM -> NEO)
// DM producer posts tiles. Two NEO TRISC0s consume immediately; two hold their tiles
// unacked until the host has logged the mismatch, then ack so the producer can drain.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"
#include "risc_common.h"
#include "api/debug/dprint.h"
#if defined(COMPILE_FOR_DM)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(UCK_CHLKC_UNPACK)
#include "internal/tt-2xx/quasar/tensix_neo_reg.h"
#endif

constexpr uint32_t num_entries = get_arg(args::num_entries);

void kernel_main() {
#if defined(COMPILE_FOR_DM)
    // DM Producer: post tiles to DFB for all 4 NEO consumers
    DataflowBuffer dfb(dfb::tile_counter_dfb);
    for (uint32_t entry = 0; entry < num_entries; entry++) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }

#elif defined(UCK_CHLKC_UNPACK)
    // NEO TRISC0 Consumer: consume tiles from DFB
    constexpr uint32_t num_consumers_to_run = get_arg(args::num_consumers_to_run);
    constexpr uint32_t sync_flag_addr = get_arg(args::sync_flag_addr);
    uint32_t thread_idx = get_my_thread_id();
    volatile uint32_t* sync_flag = reinterpret_cast<volatile uint32_t*>(sync_flag_addr);

    // Stalled consumers leave their tiles unacked until the host has logged the mismatch.
    // They ack once released so the producer DataflowBuffer destructor can drain.
    if (thread_idx >= num_consumers_to_run) {
        while (*sync_flag != 1) {
        }
    }

    DataflowBuffer dfb(dfb::tile_counter_dfb);
    for (uint32_t entry = 0; entry < num_entries; entry++) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }

    // Running consumers hold the drained state until the host has logged the stalled counters.
    if (thread_idx < num_consumers_to_run) {
        while (*sync_flag != 1) {
        }
    }
#endif
}
