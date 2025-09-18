// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr bool SRC_IS_DRAM = get_compile_time_arg_val(CTA_BASE + 0) == 1;
    constexpr uint32_t NUM_PAGES = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    const uint32_t src_base = get_arg_val<uint32_t>(0);

    const auto src_acc = TensorAccessor(ta_args, /*bank_base=*/src_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < NUM_PAGES; ++i) {
        cb_reserve_back(CB_ID, 1);
        uint32_t l1_dst = get_write_ptr(CB_ID);

        uint64_t src_noc = src_acc.get_noc_addr(i);
        noc_async_read(src_noc, l1_dst, PAGE_SIZE);
        noc_async_read_barrier();

        cb_push_back(CB_ID, 1);
    }
}
