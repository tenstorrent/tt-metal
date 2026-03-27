// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t cb_id_out0 = get_arg_val<uint32_t>(3);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(4);
    bool writer_only = get_arg_val<uint32_t>(5);

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(cb_id_out0);
    uint32_t ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;
#else
    experimental::CircularBuffer cb(cb_id_out0);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
#endif
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_dst;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#ifdef ARCH_QUASAR
        if (writer_only == false) {
            dfb.wait_front(ublock_size_tiles);
        }
        noc.async_write(dfb, dram_dst, ublock_size_bytes, {}, {.bank_id = dst_dram_bank_id, .addr = dst_addr});

        noc.async_write_barrier();
        if (writer_only == false) {
            dfb.pop_front(ublock_size_tiles);
        }
#else
        if (writer_only == false) {
            cb.wait_front(ublock_size_tiles);
        }
        noc.async_write(cb, dram_dst, ublock_size_bytes, {}, {.bank_id = dst_dram_bank_id, .addr = dst_addr});

        noc.async_write_barrier();
        if (writer_only == false) {
            cb.pop_front(ublock_size_tiles);
        }
#endif
        dst_addr += ublock_size_bytes;
    }
}
