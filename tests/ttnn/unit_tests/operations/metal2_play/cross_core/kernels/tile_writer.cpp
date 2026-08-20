// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Drains `recv` to DRAM. Ordinary local consumer -- it cannot tell that a remote core filled it.

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
    const auto acc_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_recv.get_tile_size();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_recv.wait_front(1);
        noc.async_write(dfb_recv, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = first_page + t});
        noc.async_write_barrier();
        dfb_recv.pop_front(1);
    }
}
