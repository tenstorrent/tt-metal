// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dummy reader that pushes pages to a CB from DRAM.
// Works with any CB page size — tile-sized or row-sized.
// The page size is determined by the CB configuration on the host.

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_pages = get_arg_val<uint32_t>(2);
    uint32_t cb_id = get_arg_val<uint32_t>(3);
    uint32_t pages_per_push = get_arg_val<uint32_t>(4);
    uint32_t page_size = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src;
    experimental::CircularBuffer cb(cb_id);

    for (uint32_t i = 0; i < num_pages; i += pages_per_push) {
        uint32_t curr_push = (num_pages - i < pages_per_push) ? (num_pages - i) : pages_per_push;
        uint32_t curr_bytes = page_size * curr_push;
        cb.reserve_back(curr_push);
        noc.async_read(dram_src, cb, curr_bytes, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        cb.push_back(curr_push);
        src_addr += curr_bytes;
    }
}
