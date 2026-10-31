// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fills one entry of DFB `out` with `pattern + word index` and hands it to the consumer. The host
// reads the entry back at the address it placed the DFB, so a launch whose DFB config pointed the
// kernel anywhere else leaves the pattern missing there.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t pattern = get_arg(args::pattern);

    DataflowBuffer dfb(dfb::out);
    dfb.reserve_back(1);
    // On Quasar DM this is the uncached alias, so the stores reach L1 without a cache flush.
    auto* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    const uint32_t num_words = dfb.get_entry_size() / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; i++) {
        entry[i] = pattern + i;
    }
    dfb.push_back(1);
}
