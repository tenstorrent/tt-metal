// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) intra-tensix self-loop compute kernel.
// Parallel to ../dfb_t6_intra.cpp — uses named CTAs.
//
// One DFB on a Tensix cluster, binding the same DFB as PRODUCER ("out") and
// CONSUMER ("in") on this kernel: M2 infers tensix_scope=INTRA. PACK TRISC
// increments each word by 1 before push_back; UNPACK TRISC increments by 1
// before pop_front. Net per-word delta = +2.
//
// Both bindings must use DISTINCT accessor_names ("out" / "in"), even
// though they resolve to the same DFB — M2 maps duplicate names oddly for INTRA
// (only one Neo's slice gets touched). Reference dfb::out only.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Both PRODUCER (out) and CONSUMER (in) bindings resolve to the same DFB.
    DataflowBuffer dfb(dfb::out);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    for (uint32_t i = 0; i < entries_per_neo; ++i) {
        dfb.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb.push_back(1);

        dfb.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb.pop_front(1);
    }
    dfb.finish();
}
