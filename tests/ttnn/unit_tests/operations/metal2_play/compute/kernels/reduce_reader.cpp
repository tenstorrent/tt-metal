// SPDX-License-Identifier: Apache-2.0
// PROBE A (reader half): drive the *dataflow* reduce-scaler helper entirely from a dfb:: token,
// including in TEMPLATE-PARAMETER position (`dfb_id` is a `uint32_t` non-type template param).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // dfb::scaler is a DFBBindingToken; this is the NTTP-position test.
    // NOTE: PoolType/ReduceDim must be ckernel::-qualified here. A *compute* kernel gets them
    // unqualified for free; a dataflow kernel does not.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

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
