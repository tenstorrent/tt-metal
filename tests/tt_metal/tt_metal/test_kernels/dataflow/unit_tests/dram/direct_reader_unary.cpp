// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t src_addr = get_arg_val<uint32_t>(0);     // global base address
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);  // data is in one bank
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    // DRAM page stride: callers pass Buffer::aligned_page_size() so the
    // kernel can advance through DRAM pages even when the allocator rounds
    // page_size up for alignment.
    uint32_t dram_page_stride = get_arg_val<uint32_t>(3);

    constexpr uint32_t ublock_size_tiles = 1;

    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;
    AllocatorBank<bank_type> src_dram;
    Noc noc;

    CircularBuffer cb(cb_id);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.reserve_back(ublock_size_tiles);
        noc.async_read(src_dram, cb, ublock_size_bytes, {.bank_id = src_bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        cb.push_back(ublock_size_tiles);
        src_addr += dram_page_stride;
    }
}
