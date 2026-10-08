// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 writer for test_eltwise_add_quasar: one thread writes tile i of dfb::out to page i of the
// DRAM-interleaved output tensor C, for i in [0, num_tiles). Explicit sync, as on Blackhole.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t out_tile_bytes = dfb_out.get_entry_size();

    const auto c = TensorAccessor(tensor::c);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, c, out_tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }

    dfb_out.finish();
}
