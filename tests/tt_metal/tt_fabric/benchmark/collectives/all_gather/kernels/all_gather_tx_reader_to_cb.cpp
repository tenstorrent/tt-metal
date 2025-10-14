// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"
#include "debug/dprint.h"

// CT args layout:
//   0: SRC_IS_DRAM
//   1: NUM_PAGES
//   2: PAGE_SIZE
//
// RT args:
//   0: src_base

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA = ta_args.next_compile_time_args_offset();
    constexpr bool SRC_IS_DRAM = get_compile_time_arg_val(CTA + 0) == 1;
    constexpr uint32_t NUM_PAGES = get_compile_time_arg_val(CTA + 1);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA + 2);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    constexpr uint32_t GROUP_PAGES = 4;

    const uint32_t src_base = get_arg_val<uint32_t>(0);
    const auto src_acc = TensorAccessor(ta_args, /*bank_base=*/src_base, /*page_size=*/PAGE_SIZE);

    DPRINT << "reader: begin pages=" << NUM_PAGES << " page=" << PAGE_SIZE << " src_base=0x" << src_base
           << " dram=" << (int)SRC_IS_DRAM << ENDL();

    uint32_t sent = 0;
    while (sent < NUM_PAGES) {
        uint32_t remaining = NUM_PAGES - sent;
        uint32_t this_group = remaining < GROUP_PAGES ? remaining : GROUP_PAGES;

        cb_reserve_back(CB_ID, this_group);
        uint32_t l1_base = get_write_ptr(CB_ID);
        DPRINT << "reader: reserve group=" << this_group << " l1_base=0x" << l1_base << " sent=" << sent << ENDL();

        for (uint32_t i = 0; i < this_group; ++i) {
            uint64_t src_noc = src_acc.get_noc_addr(sent + i);
            uint32_t l1_dst = l1_base + i * PAGE_SIZE;
            // Log the first page of each group (and always for tiny runs)
            if (i == 0 || NUM_PAGES <= 4) {
                uint32_t src_hi = (uint32_t)(src_noc >> 32);
                uint32_t src_lo = (uint32_t)(src_noc & 0xffffffffu);
                DPRINT << "reader: noc_async_read idx=" << (sent + i) << " src_noc[hi:lo]=0x" << src_hi << ":0x"
                       << src_lo << " -> l1=0x" << l1_dst << " bytes=" << PAGE_SIZE << ENDL();
            }
            noc_async_read(src_noc, l1_dst, PAGE_SIZE);
        }

        noc_async_read_barrier();
        DPRINT << "reader: barrier done for group=" << this_group << ENDL();

        cb_push_back(CB_ID, this_group);
        DPRINT << "reader: push_back group=" << this_group << ENDL();

        sent += this_group;
    }

    DPRINT << "reader: done total=" << sent << ENDL();
}
