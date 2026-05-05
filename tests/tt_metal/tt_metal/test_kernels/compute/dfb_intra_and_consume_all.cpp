// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combined compute kernel for the intra-tensix + remapper strided x blocked co-existence test.
//
// Two DFBs live in the same program on one Tensix cluster (N Neo threads):
//
//   dfb(0) - DM->Tensix, 1 strided producer x N blocked consumers (uses remapper).
//            Produced by a concurrent DM kernel; this kernel acts as the blocked consumer.
//
//   dfb(1) - Intra-tensix, N packer-producer -> N unpacker-consumer pairs (hidden TCs, no remapper).
//            This kernel acts as both producer and consumer for dfb(1).
//
// Execution order (all 4 TRISCs per Neo run kernel_main concurrently):
//
//   Phase 1 - blocked consumer of dfb(0):
//     UNPACK TRISC: wait_front / pop_front for every entry (active).
//     PACK / MATH TRISCs: wait_front and pop_front are no-ops; they race through Phase 1.
//
//   Phase 2 - intra-tensix on dfb(1):
//     PACK TRISC:   reserve_back -> increment words in entry (+1) -> push_back
//                   [then wait_front / pop_front are no-ops for PACK]
//     UNPACK TRISC: [reserve_back / push_back are no-ops for UNPACK]
//                   wait_front -> increment words in entry (+1) -> pop_front
//
// Net result for dfb(1) L1 ring: every word = original_value + 2.
//
// CTA layout:
//   [0] num_entries_consumer - entries each UNPACK TRISC consumes from dfb(0) (BLOCKED: all entries)
//   [1] entries_per_neo      - dfb(1) loop count per Neo (= total_entries_dfb1 / num_neos)
//   [2] words_per_entry      - words per dfb(1) entry for in-place increment

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    const uint32_t num_entries_consumer = get_compile_time_arg_val(0);
    const uint32_t entries_per_neo      = get_compile_time_arg_val(1);
    const uint32_t words_per_entry      = get_compile_time_arg_val(2);

    // Phase 1: consume all entries from the DM-produced strided x blocked DFB.
    // UNPACK TRISC: each Neo's unpacker waits for its own TC credit (remapper fans out 1 DM post
    // to N consumer TCs so every blocked UNPACK TRISC sees the full set of entries).
    // PACK / MATH TRISCs: wait_front and pop_front are no-ops; they exit Phase 1 immediately.
    experimental::DataflowBuffer dfb_consumer(0);
    for (uint32_t i = 0; i < num_entries_consumer; i++) {
        dfb_consumer.wait_front(1);
        dfb_consumer.pop_front(1);
    }
    dfb_consumer.finish();

    // Phase 2: intra-tensix DFB credit flow (packer -> unpacker, hidden TC per Neo).
    // PACK TRISC fills its Neo's TC slot; UNPACK TRISC drains it.
    // Because PACK races through Phase 1 (no-ops), it can pre-fill the entire TC capacity
    // of dfb(1) before the UNPACK TRISC arrives from Phase 1. PACK then blocks in finish()
    // until UNPACK has consumed all entries and called its own finish().
    experimental::DataflowBuffer dfb_intra(1);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    for (uint32_t i = 0; i < entries_per_neo; i++) {
        // PACK: wait for a free ring slot, increment the entry in-place, post credit.
        // UNPACK / MATH: reserve_back and push_back are no-ops.
        dfb_intra.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.push_back(1);

        // UNPACK: wait for credit, increment the entry in-place, pop credit.
        // PACK / MATH: wait_front and pop_front are no-ops.
        dfb_intra.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.pop_front(1);
    }
    dfb_intra.finish();
}
