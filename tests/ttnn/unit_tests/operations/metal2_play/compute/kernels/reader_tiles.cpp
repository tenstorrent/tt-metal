// SPDX-License-Identifier: Apache-2.0
// Generic Metal 2.0 tile reader: streams `num_tiles` pages of tensor::src into dfb::in_tiles.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    Noc noc;
    const auto acc = TensorAccessor(tensor::src);
    const uint32_t tile_bytes = in.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.reserve_back(1);
        noc.async_read(acc, in, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        in.push_back(1);
    }
}
