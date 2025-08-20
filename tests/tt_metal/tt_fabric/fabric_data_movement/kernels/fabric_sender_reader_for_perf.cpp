// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool SRC_IS_DRAM = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t NUM_PAGES = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    const uint32_t src_base = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<SRC_IS_DRAM> src_ag = {.bank_base_address = src_base, .page_size = PAGE_SIZE};

    for (uint32_t i = 0; i < NUM_PAGES; ++i) {
        cb_reserve_back(CB_ID, 1);
        uint32_t l1_dst = get_write_ptr(CB_ID);

        uint64_t src_noc = get_noc_addr(i, src_ag);
        noc_async_read(src_noc, l1_dst, PAGE_SIZE);
        noc_async_read_barrier();

        cb_push_back(CB_ID, 1);
    }
}
