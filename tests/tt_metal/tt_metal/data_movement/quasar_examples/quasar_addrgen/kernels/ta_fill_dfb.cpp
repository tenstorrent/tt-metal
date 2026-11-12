// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pattern producer for the write-only rows of the TensorAccessor AddrGen pattern matrix.
// Fills one dfb::out entry per page with a pattern that encodes the logical page id, so the host
// can check that page p landed at logical page p of the destination tensor (catches a writer that
// walks pages in the wrong order, which a symmetric read+write copy would hide).
//
// Word w of page p = (p << 16) | w. Must match fill_pattern_word() in test_tensor_accessor_addrgen.cpp.
//
// Explicit sync on purpose: entries are filled with CPU stores, not NoC reads, so there is no
// transaction for the DFB implicit-sync path to track. The consumer side still uses implicit sync.
//
// Named RTAs:
//   start_page, num_pages

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    DataflowBuffer dfb(dfb::out);
    const uint32_t words_per_entry = dfb.get_entry_size() / sizeof(uint32_t);

    for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
        dfb.reserve_back(1);
        {
            auto lock = dfb.scoped_write_lock();
            auto mem = lock.get_ptr();
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                mem[w] = (page_id << 16) | w;
            }
        }
        dfb.push_back(1);
    }
    dfb.finish();
}
