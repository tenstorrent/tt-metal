// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Both PRODUCER ("out") and CONSUMER ("in") bindings on this kernel reference
    // the same self-looped DFB, so dfb::out and dfb::in resolve to the same ID.
    DataflowBuffer dfb(dfb::out);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    for (uint32_t i = 0; i < entries_per_neo; i++) {
        // Pack TRISC: wait for free space, increment entry in-place, post credit.
        dfb.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb.push_back(1);

        // Unpack TRISC: wait for credit, increment entry in-place, consume credit.
        dfb.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        {
            if (trisc_id == 0) {
                volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_read_ptr() << 4);
                for (uint32_t w = 0; w < words_per_entry; w++) {
                    entry[w] += 1;
                }
            }
        }
#endif
        dfb.pop_front(1);
    }
    dfb.finish();
}
