// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    const uint32_t entries_per_neo = get_compile_time_arg_val(0);
    const uint32_t words_per_entry = get_compile_time_arg_val(1);

    experimental::DataflowBuffer dfb(0);

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
