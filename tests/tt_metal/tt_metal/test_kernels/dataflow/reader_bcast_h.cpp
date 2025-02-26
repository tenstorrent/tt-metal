// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(4);
    // skip arg 5 for compat with reader_diff_lengths
    uint32_t NCHtWt = get_arg_val<uint32_t>(6);
    uint32_t NC = get_arg_val<uint32_t>(7);
    uint32_t Ht = get_arg_val<uint32_t>(8);
    uint32_t Wt = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i1 = 0;
    for (uint32_t i = 0; i < NCHtWt; i += onetile) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read(src0_noc_addr, l1_write_addr_in0, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
        src0_addr += tile_bytes;

        // for each W-tile of the first tensor we push one tile from the second arg tile list
        // but we loop the second list around
        cb_reserve_back(cb_id_in1, onetile);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, onetile);
        i1++;
        src1_addr += tile_bytes;
        if (i1 == Wt) {
            // wrap around
            i1 = 0;
            src1_addr = get_arg_val<uint32_t>(4);
        }
    }
}
