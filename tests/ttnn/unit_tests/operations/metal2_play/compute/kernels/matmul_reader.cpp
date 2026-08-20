// SPDX-License-Identifier: Apache-2.0
// Reader for the matmul_block probe: two tensors -> two DFBs.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    Noc noc;
    const auto acc0 = TensorAccessor(tensor::a);
    const auto acc1 = TensorAccessor(tensor::b);
    const uint32_t tile_bytes = in0.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in0.reserve_back(1);
        in1.reserve_back(1);
        noc.async_read(acc0, in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(acc1, in1, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        in0.push_back(1);
        in1.push_back(1);
    }
}
