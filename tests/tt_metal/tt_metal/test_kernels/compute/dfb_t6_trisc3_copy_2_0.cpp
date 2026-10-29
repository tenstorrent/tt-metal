// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// trisc3 as both endpoints of a compute kernel: it pops entries from dfb::in and pushes them into
// dfb::out, copying each entry with RISC loads and stores.
// trisc0, trisc1 and trisc2 own neither DFB, so their DFB calls below return at once.
//
// Bindings (set by host KernelSpec): dfb::in CONSUMER and dfb::out PRODUCER, both with use_trisc3.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t words_per_entry = dfb_in.get_entry_size() / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        if constexpr (ckernel::TRISC_ID == 3) {
            // Both waits poll the tile counters from this RISC, so the input entry is in L1 and the output
            // entry is free once they return. get_read_ptr / get_write_ptr are in 16 B units.
            const volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dfb_in.get_read_ptr() << 4);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_out.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                dst[w] = src[w];
            }
            asm volatile("fence" ::: "memory");  // the copy lands before push_back posts the credit
        }
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
    dfb_in.finish();
    dfb_out.finish();
}
