// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(0);     // global base address
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);  // data is in one bank
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    // DRAM page stride: callers pass Buffer::aligned_page_size() so the
    // kernel can advance through DRAM pages even when the allocator rounds
    // page_size up for alignment.
    uint32_t dram_page_stride = get_arg_val<uint32_t>(3);

    uint32_t ublock_size_tiles = 1;

    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;
    AllocatorBank<bank_type> dst_dram;
    Noc noc;

    CircularBuffer cb(cb_id);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.wait_front(ublock_size_tiles);
        noc.async_write(cb, dst_dram, ublock_size_bytes, {}, {.bank_id = dst_bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dst_addr += dram_page_stride;
    }
}
