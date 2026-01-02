// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

//
// Reader (sender-side) kernel — batched DRAM→L1 copies into CB.
//
// What this does:
// - Pulls tensor pages from the source buffer into the local L1 Circular Buffer (CB c_0).
// - Works in small groups (4 pages): reserve CB space, queue N async reads, wait once, then publish N pages.
// - Grouping avoids a barrier per page and lets the writer drain the CB while we fetch the next group.
//
// How it works (per group):
//   1) cb_reserve_back(CB_ID, N) + get_write_ptr() → reserve N pages in CB and get an L1 base pointer.
//   2) Issue N noc_async_read() calls back-to-back into that reserved L1 range.
//   3) noc_async_read_barrier() → wait until all N reads complete.
//   4) cb_push_back(CB_ID, N) → make the N pages visible to the writer.
//
// Notes:
// - The CB lives in L1. The host sizes it large enough (8 pages) so reader and writer can overlap.
// - NUM_PAGES and PAGE_SIZE come from compile-time args. src_base comes from runtime args.
//

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    const uint32_t src_base = get_arg_val<uint32_t>(0);
    const uint32_t num_cb_total_pages = get_arg_val<uint32_t>(1);
    const auto src_acc = TensorAccessor(ta_args, /*bank_base=*/src_base, /*page_size=*/PAGE_SIZE);

    // read single page into CB
    cb_reserve_back(CB_ID, 1);
    uint32_t l1_dst = get_write_ptr(CB_ID);
    uint64_t src_noc = src_acc.get_noc_addr(0);
    noc_async_read(src_noc, l1_dst, PAGE_SIZE);
    noc_async_read_barrier();
    cb_push_back(CB_ID, 1);
}
