// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        cb_reserve_back(cb_id_in1, ublock_size_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
        cb_push_back(cb_id_in1, ublock_size_tiles);

        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }

// This input populates dest with values before binary operation
// executes, this is used to test eltwise binary with dest reuse
// and eltwise binary with dest accumulation
#if defined(DST_ACCUM_MODE) || defined(ELTWISE_DEST_REUSE_TYPE)
    uint32_t src2_addr = get_arg_val<uint32_t>(5);
    uint32_t src2_bank_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in2 = 2;
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id_in2);
    uint32_t l1_write_addr_in2;

    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src2_noc_addr = get_noc_addr_from_bank_id<true>(src2_bank_id, src2_addr);

        cb_reserve_back(cb_id_in2, ublock_size_tiles);

        l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        noc_async_read(src2_noc_addr, l1_write_addr_in2, ublock_size_bytes_2);
        noc_async_read_barrier();

        cb_push_back(cb_id_in2, ublock_size_tiles);

        src2_addr += ublock_size_bytes_2;
    }
#endif
}
