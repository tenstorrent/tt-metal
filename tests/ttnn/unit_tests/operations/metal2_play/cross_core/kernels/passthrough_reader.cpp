// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    const auto acc_in = TensorAccessor(tensor::in);
    const uint32_t tile_bytes = dfb_recv.get_tile_size();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_recv.reserve_back(1);
        noc.async_read(acc_in, dfb_recv, tile_bytes, {.page_id = first_page + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_recv.push_back(1);
    }
}
