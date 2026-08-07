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
// Every TRISC must call wait_front / copy_tile / pop_front in the same loop iteration.
// copy_tile self-gates per-TRISC (UNPACK=llk_unpack_A, MATH=llk_math_datacopy); only UNPACK
// actually blocks in wait/pop, but MATH must still execute copy_tile in program order after
// wait_front. Splitting PACK out or gating wait_front to UNPACK-only desyncs MATH from UNPACK
// and traps TRISC1 (ERROR_TRISC1 / 0x19).
//
// Net result for dfb(1) L1 ring: every word = original_value + 2.
//
// CTA layout:
//   [0] num_entries_consumer - entries each UNPACK TRISC consumes from dfb(0) (BLOCKED: all entries)
//   [1] entries_per_neo      - dfb(1) loop count per Neo (= total_entries_dfb1 / num_neos)
//   [2] words_per_entry      - words per dfb(1) entry for in-place increment

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_consumer = get_arg(args::num_entries_consumer);
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Phase 1: blocked consumer of the DM-produced strided x blocked DFB.
    DataflowBuffer dfb_consumer(dfb::remapper_in);

    unary_op_init_common(dfb::remapper_in, dfb::remapper_in);

    for (uint32_t i = 0; i < num_entries_consumer; i++) {
        acquire_dst();
        dfb_consumer.wait_front(1);
        copy_tile(dfb::remapper_in, 0, 0);
        dfb_consumer.pop_front(1);
        release_dst();
    }
    dfb_consumer.finish();

    // Phase 2: intra-tensix credit flow (packer -> unpacker, hidden TC per Neo).
    DataflowBuffer dfb_intra(dfb::intra_out);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    unary_op_init_common(dfb::intra_out, dfb::intra_out);

    for (uint32_t i = 0; i < entries_per_neo; i++) {
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

        acquire_dst();
        dfb_intra.wait_front(1);
        copy_tile(dfb::intra_out, 0, 0);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.pop_front(1);
        release_dst();
    }
    dfb_intra.finish();
}
