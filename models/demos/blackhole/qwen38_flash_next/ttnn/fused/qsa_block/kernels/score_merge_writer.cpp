// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge writer: per row the masked 2 KB chunk into the output row.  CB 16.
// Compile-time args: TensorAccessorArgs out ([1, 1, rows, W] ROW_MAJOR).  Runtime args: 0 out address, 1 rows, 2 first
// chunk, 3 chunks on this core.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t first_chunk = get_arg_val<uint32_t>(2);
    const uint32_t chunks = get_arg_val<uint32_t>(3);
    constexpr uint32_t CB_OUT = 16, CHUNK_BYTES = tile_rows::TILE_BYTES;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
#ifdef QSA_LIVE
    // QSA_LIVE: args 2, 3 = first item and item count (item i = chunk i / live_rows, row i % live_rows), arg 4 live
    // rows. The core of a chunk's row 0 also writes that chunk's dead rows (live_rows .. rows-1) as zeros from the zero
    // tile (CB 1, pushed by the reader).
    const uint32_t live_rows = get_arg_val<uint32_t>(4);
    for (uint32_t item = first_chunk; item < first_chunk + chunks; ++item) {
        FUSED_ZONE("fz_qs_sm_w_row");
        const uint32_t chunk = item / live_rows;
        const uint32_t r = item - chunk * live_rows;
        cb_wait_front(CB_OUT, 1);
        noc_async_write(get_read_ptr(CB_OUT), out.get_noc_addr(r, chunk * CHUNK_BYTES), CHUNK_BYTES);
        if (r == 0 && live_rows < rows) {
            cb_wait_front(1, 1);
            for (uint32_t dead = live_rows; dead < rows; ++dead) {
                noc_async_write(get_read_ptr(1), out.get_noc_addr(dead, chunk * CHUNK_BYTES), CHUNK_BYTES);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
#else
    for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {
        for (uint32_t r = 0; r < rows; ++r) {
            FUSED_ZONE("fz_qs_sm_w_row");
            cb_wait_front(CB_OUT, 1);
            noc_async_write(get_read_ptr(CB_OUT), out.get_noc_addr(r, chunk * CHUNK_BYTES), CHUNK_BYTES);
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, 1);
        }
    }
#endif
}
