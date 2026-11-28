// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// For test_gguf_decode_mm.cpp: all of c_0's pages in one read (pages of any size), then c_1's tiles one by one.
// Runtime args: page buffer address, page bytes in all, in0 buffer address, K tiles.

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    const uint32_t page_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_bytes = get_arg_val<uint32_t>(1);
    uint32_t in0_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_tiles = get_arg_val<uint32_t>(3);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    CircularBuffer pages(tt::CBIndex::c_0);
    CircularBuffer in0(tt::CBIndex::c_1);
    pages.reserve_back(k_tiles);
    noc.async_read(dram, pages, page_bytes, {.bank_id = 0, .addr = page_addr}, {});
    noc.async_read_barrier();
    pages.push_back(k_tiles);
    const uint32_t in0_bytes = in0.get_tile_size();
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        in0.reserve_back(1);
        noc.async_read(dram, in0, in0_bytes, {.bank_id = 0, .addr = in0_addr}, {});
        noc.async_read_barrier();
        in0.push_back(1);
        in0_addr += in0_bytes;
    }
}
