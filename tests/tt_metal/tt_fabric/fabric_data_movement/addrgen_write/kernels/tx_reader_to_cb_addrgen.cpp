// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"
#include "kernel_common.hpp"

//
// Unified reader (sender-side) kernel — consolidates 4 variants.
// Pulls tensor pages from the source buffer into the local L1 Circular Buffer (CB c_0).
// Works in small groups: reserve CB space, queue N async reads, wait once, then publish N pages.
// Grouping avoids a barrier per page and lets the writer drain the CB while we fetch the next group.
//
// CT args:
//   0: OPERATION_TYPE (OperationType enum: BasicWrite, Scatter, FusedAtomicInc)
//   1: SRC_IS_DRAM (0=L1, 1=DRAM)
//   2: NUM_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (buffer spacing for address calculation)
//
// RT args:
//   0: src_base (u32)

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t OPERATION_TYPE = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr bool SRC_IS_DRAM = get_compile_time_arg_val(CTA_BASE + 1) == 1;
    constexpr uint32_t NUM_PAGES = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Cast to enum type for clearer comparison
    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);

    // Process pages in groups: Scatter uses 2 pages (to match writer consumption rate),
    // BasicWrite and FusedAtomicInc use 4 pages; CB capacity will be sized larger on the host.
    constexpr uint32_t GROUP_PAGES = (operation_type == OperationType::Scatter) ? 2 : 4;

    const uint32_t src_base = get_arg_val<uint32_t>(0);
    // Use ALIGNED_PAGE_SIZE for address calculation (buffer spacing)
    const auto src_acc = TensorAccessor(ta_args, /*bank_base=*/src_base, /*page_size=*/ALIGNED_PAGE_SIZE);

    uint32_t sent = 0;
    while (sent < NUM_PAGES) {
        // how many pages in this group (last group may be smaller than GROUP_PAGES)
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
            // CB is configured with ALIGNED_PAGE_SIZE, so stride by that amount
            uint32_t l1_dst = l1_base + i * ALIGNED_PAGE_SIZE;
            // But only transfer PAGE_SIZE bytes (actual data)
            noc_async_read(src_noc, l1_dst, PAGE_SIZE);
        }

        // wait for the group to complete, then publish the group to the consumer
        noc_async_read_barrier();
        cb_push_back(CB_ID, this_group);

        sent += this_group;
    }
}
