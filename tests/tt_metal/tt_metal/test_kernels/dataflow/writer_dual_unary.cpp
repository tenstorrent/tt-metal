// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"
#include "api/debug/dprint.h"

inline void write_tiles(uint32_t num_tiles, uint32_t dst_addr, uint32_t bank_id, uint32_t buffer_id_out) {
    uint32_t ublock_size_tiles = 1;

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_dst;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(buffer_id_out);
    uint32_t ublock_size_bytes = dfb.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb.wait_front(ublock_size_tiles);
        noc.async_write(dfb, dram_dst, ublock_size_bytes, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        dfb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
#else
    experimental::CircularBuffer cb(buffer_id_out);
    uint32_t ublock_size_bytes = cb.get_tile_size();

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.wait_front(ublock_size_tiles);
        noc.async_write(cb, dram_dst, ublock_size_bytes, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
#endif
}

void kernel_main() {
    uint32_t dst_addr_0 = get_arg_val<uint32_t>(0);
    uint32_t bank_id_0 = get_arg_val<uint32_t>(1);
    uint32_t dst_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t bank_id_1 = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t dst0_dfb_id =
#ifdef ARCH_QUASAR
        2;
#else
        tt::CBIndex::c_16;
#endif
    constexpr uint32_t dst1_dfb_id =
#ifdef ARCH_QUASAR
        3;
#else
        tt::CBIndex::c_17;
#endif

    write_tiles(num_tiles, dst_addr_0, bank_id_0, dst0_dfb_id);
    write_tiles(num_tiles, dst_addr_1, bank_id_1, dst1_dfb_id);
}
