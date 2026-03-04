// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#endif

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(cb_id);
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
    experimental::AllocatorBank<bank_type> dst_dram;
    experimental::Noc noc;

    uint32_t ublock_size_bytes = dfb.get_entry_size();

    volatile tt_l1_ptr uint32_t* debug_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(763520);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb.wait_front(ublock_size_tiles);

        DPRINT << "Debug pointer value: " << *debug_ptr << ENDL();

        noc.async_write(dfb, dst_dram, ublock_size_bytes, {}, {.bank_id = dst_bank_id, .addr = dst_addr});
        noc.async_write_barrier();

        dfb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
#else
    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
         uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);

        cb_wait_front(cb_id, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
#endif
}
