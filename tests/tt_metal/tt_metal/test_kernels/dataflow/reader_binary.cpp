// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks

    experimental::CircularBuffer cb0(cb_id_in0);
    experimental::CircularBuffer cb1(cb_id_in1);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src;

    uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes_0 = cb0.get_tile_size() * ublock_size_tiles;
    uint32_t ublock_size_bytes_1 = cb1.get_tile_size() * ublock_size_tiles;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

        cb0.reserve_back(ublock_size_tiles);
        cb1.reserve_back(ublock_size_tiles);
        noc.async_read(dram_src, cb0, ublock_size_bytes_0, {.bank_id = src0_bank_id, .addr = src0_addr}, {});
        noc.async_read(dram_src, cb1, ublock_size_bytes_1, {.bank_id = src1_bank_id, .addr = src1_addr}, {});
        noc.async_read_barrier();
        cb0.push_back(ublock_size_tiles);
        cb1.push_back(ublock_size_tiles);
        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }


    // This input populates dest with values before binary operation
    // executes, this is used to test eltwise binary with dest re-use
    // and eltwise binary with dest accumulation
#if defined(DST_ACCUM_MODE) || defined(ELTWISE_DEST_REUSE_TYPE)
    uint32_t src2_addr = get_arg_val<uint32_t>(5);
    uint32_t src2_bank_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in2 = 2;
    experimental::CircularBuffer cb2(cb_id_in2);
    uint32_t ublock_size_bytes_2 = cb2.get_tile_size() * ublock_size_tiles;

    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src2_noc_addr = get_noc_addr_from_bank_id<true>(src2_bank_id, src2_addr);

        cb2.reserve_back(ublock_size_tiles);
        noc.async_read(dram_src, cb2, ublock_size_bytes_2, {.bank_id = src2_bank_id, .addr = src2_addr}, {});
        noc.async_read_barrier();
        cb2.push_back(ublock_size_tiles);
        src2_addr += ublock_size_bytes_2;
    }
#endif
}
