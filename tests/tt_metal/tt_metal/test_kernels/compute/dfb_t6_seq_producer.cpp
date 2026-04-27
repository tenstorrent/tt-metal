// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential Tensix producer for concurrent DFB stress tests (Gap 7).
//
// A single Neo thread loops through num_dfbs DFBs in order, signaling entries for
// each one.  The host pre-fills every DFB's L1 ring before the program is launched.
// Running on one Neo thread avoids any dependency on Neo hartid values while still
// creating all num_dfbs DFBs in a single Program (so their TCs are allocated
// simultaneously), which is the TC-allocator stress this test exercises.
//
// Synchronisation: after each dfb.finish() the producer has verified that all
// consumers of that DFB have acked every entry, so it is safe to move on to the
// next DFB.  Concurrent DM consumers therefore drain each DFB as soon as entries
// arrive, then block on the next DFB until the Neo fills it.
//
// Compile-time args:
//   [0]: num_dfbs                  � number of DFBs to loop through
//   [1]: num_entries_per_producer  � entries to signal per DFB (same for all)

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t num_dfbs               = get_compile_time_arg_val(0);
    const uint32_t     num_entries_per_producer = get_compile_time_arg_val(1);

    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        experimental::DataflowBuffer dfb(dfb_id);
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
            dfb.reserve_back(1);
            dfb.push_back(1);
        }
        // Blocks until all consumers of this DFB have acked every entry.
        dfb.finish();
    }
}
