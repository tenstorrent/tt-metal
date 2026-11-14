// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF page helpers (experimental/gguf_page.h) on a data movement core, for test_gguf_page_helpers.cpp. Pages from
// DRAM land in c_0.
//   GGUF_LOOKUP_RVV = 0: prepare_page<GGUF_FMT, GGUF_BRANCH_FREE> on each page in place, the pages back to DRAM.
//   GGUF_LOOKUP_RVV = 1: the pages go to the compute kernel gguf_lookup_rvv.cpp; its two results (c_16 scalar, c_17
//                        vector) go back to DRAM.
// Runtime args: src address, dst address, dst2 address (RVV only), page count. DRAM bank 0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/gguf_page.h"

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t dst2 = get_arg_val<uint32_t>(2);
    const uint32_t pages = get_arg_val<uint32_t>(3);
    constexpr auto FMT = static_cast<experimental::gguf::Format>(GGUF_FMT);
    constexpr uint32_t page_bytes = experimental::gguf::page_layout<FMT>().rows * 64;
    constexpr uint32_t cb_pages = 0;

    cb_reserve_back(cb_pages, pages);
    const uint32_t l1 = get_write_ptr(cb_pages);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, src), l1, pages * page_bytes);
    noc_async_read_barrier();
#if GGUF_LOOKUP_RVV
    cb_push_back(cb_pages, pages);
    const uint32_t outs[2] = {dst, dst2};
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t cb_out = 16 + i;
        cb_wait_front(cb_out, pages);
        noc_async_write(get_read_ptr(cb_out), get_noc_addr_from_bank_id<true>(0, outs[i]), pages * page_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, pages);
    }
#else
    (void)dst2;
    for (uint32_t p = 0; p < pages; p++) {
        experimental::gguf::prepare_page<FMT, GGUF_BRANCH_FREE>(l1 + p * page_bytes);
    }
    noc_async_write(l1, get_noc_addr_from_bank_id<true>(0, dst), pages * page_bytes);
    noc_async_write_barrier();
#endif
}
