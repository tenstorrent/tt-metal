// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t HtWt = get_arg(args::HtWt);

    DataflowBuffer dfb_out(dfb::in);

    Noc noc;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = dfb_out.get_entry_size();

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const auto s = TensorAccessor(tensor::dst_tensor);

    // this writer will write a NWH stream back to a NHW tensor (inverse of reader_unary_transpose_wh_8bank)
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                dfb_out.wait_front(onetile);

                noc.async_write(dfb_out, s, tile_bytes, {}, {.page_id = i_tile});
                noc.async_write_barrier();

                dfb_out.pop_front(onetile);
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
