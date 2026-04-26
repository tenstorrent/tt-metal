// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Simple reader that pushes `total_tiles` from DRAM into c_24.
// Used by reblock_and_untilize and add_bias_bcast_rows isolated tests.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);
    uint32_t total_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = tt::CBIndex::c_24;

    uint64_t src_noc = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

    cb_reserve_back(cb_id, total_tiles);
    uint32_t l1_w = get_write_ptr(cb_id);
    noc_async_read(src_noc, l1_w, total_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id, total_tiles);
}
