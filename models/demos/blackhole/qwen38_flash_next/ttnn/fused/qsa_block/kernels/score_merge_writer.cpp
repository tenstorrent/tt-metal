// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge writer: per row the masked 2 KB chunk into the output row.  CB 16.
// Compile-time args: TensorAccessorArgs out ([1, 1, rows, W] ROW_MAJOR).  Runtime args: 0 out address, 1 rows, 2 first
// chunk, 3 chunks on this core.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t first_chunk = get_arg_val<uint32_t>(2);
    const uint32_t chunks = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = 16, chunk_bytes = tile_rows::tile_bytes;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {
        for (uint32_t r = 0; r < rows; ++r) {
            CircularBuffer(cb_out).wait_front(1);
            noc.async_write(
                CircularBuffer(cb_out), out, chunk_bytes, {}, {.page_id = r, .offset_bytes = chunk * chunk_bytes});
            noc.async_write_barrier();
            CircularBuffer(cb_out).pop_front(1);
        }
    }
}
