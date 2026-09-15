// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_count = get_compile_time_arg_val(1);
    constexpr auto accessor_args = TensorAccessorArgs<2>();
    auto out = TensorAccessor(accessor_args, get_arg_val<uint32_t>(0));
    DataflowBuffer cb(16);
    Noc noc;
    const uint32_t bytes = get_tile_size(16);
    for (uint32_t tile = 0; tile < tile_count; tile += batch) {
        cb.wait_front(batch);
        // Steady state is CB drain only: no DRAM or L1 copy. Retain exactly
        // the final batch for correctness, then return its slots to PACK.
        if (tile + batch == tile_count) {
            for (uint32_t j = 0; j < batch; ++j) {
                noc.async_write(cb, out, bytes, {.offset_bytes = j * bytes}, {.page_id = j});
            }
            noc.async_write_barrier();
        }
        cb.pop_front(batch);
    }
}
