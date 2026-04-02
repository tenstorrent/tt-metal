// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/noc.h"
#include "experimental/tensor.h"

// #include "api/debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4);  // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);

    constexpr uint32_t in_id = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_in(in_id);
#else
    experimental::CircularBuffer cb(in_id);
#endif

    experimental::Noc noc;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
#ifdef ARCH_QUASAR
    uint32_t tile_bytes = dfb_in.get_entry_size();
#else
    uint32_t tile_bytes = cb.get_tile_size();
#endif

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
#ifdef ARCH_QUASAR
                dfb_in.reserve_back(onetile);

                noc.async_read(s, dfb_in, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();

                dfb_in.push_back(onetile);
#else
                cb.reserve_back(onetile);

                noc.async_read(s, cb, tile_bytes, {.page_id = i_tile}, {});
                noc.async_read_barrier();

                cb.push_back(onetile);
#endif
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
