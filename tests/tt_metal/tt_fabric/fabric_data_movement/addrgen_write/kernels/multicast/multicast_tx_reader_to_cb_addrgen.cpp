// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

//
// Reader (sender-side) kernel — batched DRAM→L1 copies into CB.
//
// What this does:
// - Pulls tensor pages from the source buffer into the local L1 Circular Buffer (CB c_0).
// - Works in small groups (4 pages): reserve CB space, queue N async reads, wait once, then publish N pages.
// - Grouping avoids a barrier per page and lets the writer drain the CB while we fetch the next group.
//

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr bool SRC_IS_DRAM = get_compile_time_arg_val(CTA_BASE + 0) == 1;
    constexpr uint32_t NUM_PAGES = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Process pages in groups of 4; CB capacity will be sized larger on the host.
    constexpr uint32_t GROUP_PAGES = 4;

    const uint32_t src_base = get_arg_val<uint32_t>(0);
    const auto src_acc = TensorAccessor(ta_args, /*bank_base=*/src_base, /*page_size=*/PAGE_SIZE);

    uint32_t sent = 0;
    while (sent < NUM_PAGES) {
        // how many pages in this group (last group may be smaller than 4)
        uint32_t this_group = GROUP_PAGES;
        uint32_t remaining = NUM_PAGES - sent;
        if (remaining < this_group) {
            this_group = remaining;
        }

        // reserve space for whole group and get base write pointer
        cb_reserve_back(CB_ID, this_group);
        uint32_t l1_base = get_write_ptr(CB_ID);

        // queue all reads for the group
        for (uint32_t i = 0; i < this_group; ++i) {
            uint64_t src_noc = src_acc.get_noc_addr(sent + i);
            uint32_t l1_dst = l1_base + i * PAGE_SIZE;
            noc_async_read(src_noc, l1_dst, PAGE_SIZE);
        }

        // wait for the group to complete, then publish the group to the consumer
        noc_async_read_barrier();
        cb_push_back(CB_ID, this_group);

        sent += this_group;
    }
}
