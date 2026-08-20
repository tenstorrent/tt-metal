// SPDX-License-Identifier: Apache-2.0
// Generic Metal 2.0 tile writer: drains dfb::out_tiles into tensor::dst.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer out(dfb::out_tiles);
    Noc noc;
    const auto acc = TensorAccessor(tensor::dst);
    const uint32_t tile_bytes = out.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        out.wait_front(1);
        noc.async_write(out, acc, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        out.pop_front(1);
    }
}
