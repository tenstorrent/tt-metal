// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) variant of compute/dfb_intra_and_consume_all.cpp.
//
// Two DFBs live in the same program on one Tensix cluster (num_threads Neos):
//
//   dfb::consume - DM->Tensix, 1 strided producer x N blocked consumers (remapper).
//                  This kernel acts as the blocked consumer; data is produced by
//                  a concurrent DM kernel.
//
//   dfb::intra   - Intra-tensix, N packer-producer -> N unpacker-consumer pairs.
//                  This kernel binds the same DFB as PRODUCER + CONSUMER, which
//                  is how M2 infers tensix_scope=INTRA.
//
// Net result for dfb::intra L1 ring: every word = original_value + 2.

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_consumer = get_arg(args::num_entries_consumer);
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Phase 1: blocked consumer of the DM-produced strided x blocked DFB.
    // Remapper fans out 1 DM post to N UNPACK TC slots so every blocked UNPACK
    // TRISC sees the full set of entries. PACK/MATH TRISCs race through
    // (wait_front / pop_front are no-ops on those TRISCs).
    DataflowBuffer dfb_consumer(dfb::consume);
    for (uint32_t i = 0; i < num_entries_consumer; ++i) {
        dfb_consumer.wait_front(1);
        dfb_consumer.pop_front(1);
    }
    dfb_consumer.finish();

    // Phase 2: intra-tensix credit flow (PACK -> UNPACK on the same Neo, hidden TC).
    DataflowBuffer dfb_intra(dfb::intra);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    for (uint32_t i = 0; i < entries_per_neo; ++i) {
        dfb_intra.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.push_back(1);

        dfb_intra.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.pop_front(1);
    }
    dfb_intra.finish();
}
