// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TRISC3 (isolated SFPU TRISC) DFB sanity kernel. The host puts both DFBs' compute endpoint on
// TRISC3 (DFBAdvancedOptions::compute_endpoint_on_trisc3), so only TRISC3 runs the sync calls;
// on UNPACK, MATH and PACK they do nothing. TRISC3 copies each entry of dfb::in into dfb::out
// with RISC loads and stores.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);

    for (uint32_t i = 0; i < num_entries; ++i) {
        in.wait_front(1);
        out.reserve_back(1);
#ifdef UCK_CHLKC_ISOLATE_SFPU
        // Read/write pointers are in 16B units.
        const volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in.get_read_ptr() << 4);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out.get_write_ptr() << 4);
        for (uint32_t w = 0; w < in.get_entry_size() / sizeof(uint32_t); ++w) {
            dst[w] = src[w];
        }
#endif
        in.pop_front(1);
        out.push_back(1);
    }
}
