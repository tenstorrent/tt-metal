// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    experimental::Noc noc;
    uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes;

    // single-tile ublocks
    #ifdef ARCH_QUASAR
        constexpr uint32_t dfb_out_id = get_compile_time_arg_val(0);
        experimental::DataflowBuffer dfb_out(dfb_out_id);
        ublock_size_bytes = dfb_out.get_entry_size();
    #else
        constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
        ublock_size_bytes = get_tile_size(cb_out_id);
        experimental::CircularBuffer cb(cb_out_id);
    #endif

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        #ifdef ARCH_QUASAR
            dfb_out.wait_front(ublock_size_tiles);
            noc.async_write(
                dfb_out,
                experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                ublock_size_bytes,
                {},
                {.bank_id = bank_id, .addr = dst_addr});
            noc.async_write_barrier();
            dfb_out.pop_front(ublock_size_tiles);
        #else
            cb.wait_front(ublock_size_tiles);
            noc.async_write(
                cb,
                experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                ublock_size_bytes,
                {},
                {.bank_id = bank_id, .addr = dst_addr});
            noc.async_write_barrier();
            cb.pop_front(ublock_size_tiles);
        #endif
        dst_addr += ublock_size_bytes;
    }
}
