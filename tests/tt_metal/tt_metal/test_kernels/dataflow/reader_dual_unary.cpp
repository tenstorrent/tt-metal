// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"
#include "experimental/noc.h"

inline void read_tiles(
    experimental::Noc& noc, uint32_t num_tiles, uint32_t src_addr, uint32_t bank_id, uint32_t buffer_id_in) {
    constexpr uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    (void)bank_id;
    experimental::DataflowBuffer dfb(buffer_id_in);
    uint32_t ublock_size_bytes = dfb.get_entry_size();
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_tensor = TensorAccessor(src_args, src_addr, ublock_size_bytes);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb.reserve_back(ublock_size_tiles);
        uint32_t l1_write_addr = dfb.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR;
        uint64_t src_noc_addr = get_noc_addr(i, src_tensor);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);
        noc.async_read_barrier();
        dfb.push_back(ublock_size_tiles);
    }
#else
    experimental::CircularBuffer cb(buffer_id_in);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.reserve_back(ublock_size_tiles);
        noc.async_read(dram_bank, cb, ublock_size_bytes, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        cb.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
#endif
}

void kernel_main() {
    uint32_t src_addr_0 = get_arg_val<uint32_t>(0);
    uint32_t bank_id_0 = get_arg_val<uint32_t>(1);
    uint32_t src_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t bank_id_1 = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t src0_dfb_id = 0;
    constexpr uint32_t src1_dfb_id = 1;

    experimental::Noc noc;

    read_tiles(noc, num_tiles, src_addr_0, bank_id_0, src0_dfb_id);
    read_tiles(noc, num_tiles, src_addr_1, bank_id_1, src1_dfb_id);
}
