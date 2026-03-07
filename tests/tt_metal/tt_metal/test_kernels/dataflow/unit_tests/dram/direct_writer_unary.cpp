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
    uint32_t dst_addr  = get_arg_val<uint32_t>(0); // global base address
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1); // data is in one bank
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    uint32_t consumer_mask = get_arg_val<uint32_t>(3);
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));
    // TODO: Replace with get_thread_idx() kernel API when available
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));

    experimental::DataflowBuffer dfb(cb_id);
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
    experimental::AllocatorBank<bank_type> dst_dram;
    experimental::Noc noc;

    uint32_t ublock_size_bytes = dfb.get_entry_size();

    uint32_t tlocal_dst_addr = dst_addr + (consumer_idx * dfb.get_stride_size());

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb.wait_front(ublock_size_tiles);

        DPRINT << "Writing to DRAM: " << tlocal_dst_addr << " from DFB" << ENDL();
        noc.async_write(dfb, dst_dram, ublock_size_bytes, {}, {.bank_id = dst_bank_id, .addr = tlocal_dst_addr});
        noc.async_write_barrier();

        dfb.pop_front(ublock_size_tiles);
        tlocal_dst_addr += dfb.get_stride_size();
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
