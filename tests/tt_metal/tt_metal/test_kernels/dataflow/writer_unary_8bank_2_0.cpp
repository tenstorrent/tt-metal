// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t onetile = 1;
    DataflowBuffer dfb_out(dfb::in);
    uint32_t tile_bytes = dfb_out.get_entry_size();
    const auto s = TensorAccessor(tensor::dst_tensor);

    Noc noc;

    for (uint32_t i = 0; i < num_tiles; i++) {
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
