// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_count = get_compile_time_arg_val(1);
    constexpr uint32_t resident_tiles = 8;
    constexpr auto accessor_args = TensorAccessorArgs<2>();
    auto src = TensorAccessor(accessor_args, get_arg_val<uint32_t>(0));
    DataflowBuffer cb(0);
    Noc noc;
    const uint32_t bytes = get_tile_size(0);

    // Initialize the WHOLE ring, including the second FP32-DST batch. There
    // are no source writes or NoC reads after this one-time initialization.
    cb.reserve_back(resident_tiles);
    for (uint32_t tile = 0; tile < resident_tiles; ++tile) {
        noc.async_read(src, cb, bytes, {.page_id = tile}, {.offset_bytes = tile * bytes});
    }
    noc.async_read_barrier();
    for (uint32_t tile = 0; tile < tile_count; tile += batch) {
        if (tile != 0) {
            cb.reserve_back(batch);
        }
        cb.push_back(batch);
    }
}
