// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes the tiles of dataflow buffer out to tiles start_id .. start_id + num_tiles - 1 of tensor out.
// With USE_VARARGS, num_tiles and start_id are runtime varargs 0 and 1 instead of named arguments.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef USE_VARARGS
    const uint32_t num_tiles = get_vararg(0);
    const uint32_t start_id = get_vararg(1);
#else
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
#endif

    DataflowBuffer dfb_out(dfb::out);
    Noc noc;
    const auto tensor_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, tensor_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }
}
