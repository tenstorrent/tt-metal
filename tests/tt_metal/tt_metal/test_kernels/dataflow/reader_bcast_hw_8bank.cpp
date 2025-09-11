// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    // skip args 1,2,5,6,7 for compat with single bank readers and reader_diff_lengths
    uint32_t NCHtWt = get_arg_val<uint32_t>(6);
    uint32_t NC = get_arg_val<uint32_t>(7);
    uint32_t Ht = get_arg_val<uint32_t>(8);
    uint32_t Wt = get_arg_val<uint32_t>(9);
    uint32_t nc1 = get_arg_val<uint32_t>(10);  // if 1 we expect the bcast tensor to have NC=1 and wrap around in NC

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;

    const auto s0 = TensorAccessor(src0_args, src0_addr, in0_tile_bytes);
    const auto s1 = TensorAccessor(src1_args, src1_addr, in1_tile_bytes);

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                uint64_t src0_noc = get_noc_addr(i, s0);
                noc_async_read(src0_noc, l1_write_addr_in0, in0_tile_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);

                // for each H,W-tile of the first tensor we push one tile from the second arg tile list
                // but we don't advance the second tile index for H,W
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                uint64_t src1_noc = get_noc_addr(i1, s1);
                noc_async_read(src1_noc, l1_write_addr_in1, in1_tile_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);

                i++;  // input tile iterates over NC Ht Wt
            }  // wt loop
        }  // ht loop
        if (nc1 == 0) {
            i1++;  // bcast-HW tile iterates only for nc loop and only if NC>1
        }
    }  // nc loop
}
