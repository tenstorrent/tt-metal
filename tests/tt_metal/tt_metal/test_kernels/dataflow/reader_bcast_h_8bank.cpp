// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    // skip args 1,2,5,6,7 for compat with single bank readers and reader_diff_lengths
    uint32_t NCHtWt = get_arg_val<uint32_t>(6);
    uint32_t NC = get_arg_val<uint32_t>(7);
    uint32_t Ht = get_arg_val<uint32_t>(8);
    uint32_t Wt = get_arg_val<uint32_t>(9);
    uint32_t nc1 = get_arg_val<uint32_t>(10);  // if 1 we expect the bcast tensor to have NC=1

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const auto s0 = TensorAccessor(src0_args, src0_addr, tile_bytes);
    const auto s1 = TensorAccessor(src1_args, src1_addr, tile_bytes);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::Noc noc;

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_in0.reserve_back(onetile);

                noc.async_read(s0, cb_in0, tile_bytes, {.page_id = i}, {});
                noc.async_read_barrier();
                cb_in0.push_back(onetile);

                // for each W-tile of the first tensor we push one tile from the second arg tile list
                // but we loop the second list around
                cb_in1.reserve_back(onetile);
                noc.async_read(s1, cb_in1, tile_bytes, {.page_id = i1}, {});
                noc.async_read_barrier();

                cb_in1.push_back(onetile);
                i1++;
                i++;  // input tile iterates over NC Ht Wt
            }

            // bcast tensor should be NC1W (actually NC32W padded with 0s in H)
            // wrap W around for each h (broadcast)
            i1 -= Wt;
        }
        // we reused Wt tiles out of NCWt bcast tensor Ht times, now advance for next NC
        if (nc1 == 0) {  // if bcast NC==1 we don't advance but reuse the tensor
            i1 += Wt;
        }
    }
}
