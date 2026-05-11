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
//   [0]: num_dfbs                  � number of DFBs to loop through (legacy)
//   [1]: num_entries_per_producer  � entries to signal per DFB (same for all)
//
// QUASAR named args:
//   args::num_dfbs                  - same as legacy CTA[0]
//   args::num_entries_per_producer  - same as legacy CTA[1]
// QUASAR compiler defines:
//   NUM_DFBS                        - matches args::num_dfbs; gates per-DFB binding
//                                     dispatch (each unrolled case references
//                                     dfb::dfb_<i> which only exists for declared
//                                     bindings).

#include "api/dataflow/dataflow_buffer.h"
#ifdef ARCH_QUASAR
#include "experimental/kernel_args.h"
#endif

#ifdef ARCH_QUASAR
#define DFB_T6_SEQ_PRODUCE(I)                                                       \
    do {                                                                            \
        DataflowBuffer dfb(dfb::dfb_##I);                                           \
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) { \
            dfb.reserve_back(1);                                                    \
            dfb.push_back(1);                                                       \
        }                                                                           \
        dfb.finish();                                                               \
    } while (0)
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

#if NUM_DFBS >= 1
    DFB_T6_SEQ_PRODUCE(0);
#endif
#if NUM_DFBS >= 2
    DFB_T6_SEQ_PRODUCE(1);
#endif
#if NUM_DFBS >= 3
    DFB_T6_SEQ_PRODUCE(2);
#endif
#if NUM_DFBS >= 4
    DFB_T6_SEQ_PRODUCE(3);
#endif
#if NUM_DFBS >= 5
    DFB_T6_SEQ_PRODUCE(4);
#endif
#if NUM_DFBS >= 6
    DFB_T6_SEQ_PRODUCE(5);
#endif
#else
    constexpr uint32_t num_dfbs               = get_compile_time_arg_val(0);
    const uint32_t     num_entries_per_producer = get_compile_time_arg_val(1);

    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        DataflowBuffer dfb(dfb_id);
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
            dfb.reserve_back(1);
            dfb.push_back(1);
        }
        // Blocks until all consumers of this DFB have acked every entry.
        dfb.finish();
    }
#endif
}
