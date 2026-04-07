// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool use_dfbs = get_compile_time_arg_val(1) == 1;
    uint32_t dst_addr  = get_arg_val<uint32_t>(0); // global base address
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1); // data is in one bank
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    uint32_t consumer_idx = get_my_thread_id();
#else
    uint32_t consumer_idx = 0;
#endif

    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
    experimental::AllocatorBank<bank_type> dst_dram;
    experimental::Noc noc;

    if constexpr (use_dfbs) {
        experimental::DataflowBuffer dfb(cb_id);
        uint32_t ublock_size_bytes = dfb.get_entry_size();

        uint32_t tlocal_dst_addr = dst_addr + (consumer_idx * ublock_size_bytes);

        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#ifdef ARCH_QUASAR
            dfb.write_out(noc, dst_dram, {.bank_id = dst_bank_id, .addr = tlocal_dst_addr});
#else
            dfb.wait_front(ublock_size_tiles);
            noc.async_write(dfb, dst_dram, ublock_size_bytes, {}, {.bank_id = dst_bank_id, .addr = tlocal_dst_addr});
            noc.async_write_barrier();
            dfb.pop_front(ublock_size_tiles);
#endif
            tlocal_dst_addr += dfb.get_stride_size();
        }

        dfb.finish();

#ifdef ARCH_QUASAR
        // TODO: This will be replaced with some dfb.commit or noc.async_write_barrier call
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[cb_id];
        for (uint32_t i = 0; i < local_dfb_interface.num_txn_ids; i++) {
            noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(local_dfb_interface.txn_ids[i]);
        }
#endif
    } else {
        experimental::CircularBuffer cb(cb_id);
        uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
            cb.wait_front(ublock_size_tiles);
            noc.async_write(cb, dst_dram, ublock_size_bytes, {}, {.bank_id = dst_bank_id, .addr = dst_addr});
            noc.async_write_barrier();
            cb.pop_front(ublock_size_tiles);
            dst_addr += ublock_size_bytes;
        }
    }
}
