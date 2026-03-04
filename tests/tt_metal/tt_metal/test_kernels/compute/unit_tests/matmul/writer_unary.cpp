// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

void kernel_main() {
    const uint32_t out_cb = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_bank_id_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // single-tile ublocks

    experimental::CircularBuffer cb(out_cb);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_dst;

    uint32_t ublock_size_bytes = cb.get_tile_size();
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.wait_front(ublock_size_tiles);

        noc.async_write(cb, dram_dst, ublock_size_bytes, {}, {.bank_id = dst_dram_bank_id_addr, .addr = dst_addr});

        noc.async_write_barrier();

        cb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
