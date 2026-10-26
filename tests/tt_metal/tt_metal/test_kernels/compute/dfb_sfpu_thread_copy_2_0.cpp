// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU-thread DFB sanity kernel. Built with DFB_SYNC_ON_SFPU_THREAD, so only the SFPU
// thread runs the sync calls; on UNPACK, MATH and PACK they do nothing. The SFPU thread copies each entry of
// dfb::in into dfb::out with RISC loads and stores.

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
        const volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in.get_read_ptr() << cb_addr_shift);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out.get_write_ptr() << cb_addr_shift);
        const uint32_t num_words = in.get_entry_size() / sizeof(uint32_t);
        for (uint32_t w = 0; w < num_words; ++w) {
            dst[w] = src[w];
        }
        // RISC L1 stores are not ordered against push_back's Tensix instruction: load back the
        // last stored word and fence, as ckernel::memcpy_blocking does, so the entry has landed.
        (void)dst[num_words - 1];
        asm volatile("fence" ::: "memory");
#endif
        in.pop_front(1);
        out.push_back(1);
    }
}
