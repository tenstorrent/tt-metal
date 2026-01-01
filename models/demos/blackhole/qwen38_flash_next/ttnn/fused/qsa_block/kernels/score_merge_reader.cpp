// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge reader: the zero tile, then per chunk of this core and per row the four devices' 2 KB chunks of the
// gathered score rows and the mask chunk.  CBs: 0 device chunks (bf16, 4), 1 zero (bf16, 1), 2 mask (bf16, 1).
// Compile-time args: TensorAccessorArgs gathered ([1, 1, 4 * rows, W] ROW_MAJOR), mask ([1, 1, rows, W]).
// Runtime args: 0 gathered, 1 mask addresses, 2 rows, 3 first chunk, 4 chunks on this core, 5 total chunks (device d
// row r chunk c at page (d * rows + r) * total + c).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"

void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t first_chunk = get_arg_val<uint32_t>(3);
    const uint32_t chunks = get_arg_val<uint32_t>(4);
    const uint32_t total_chunks = get_arg_val<uint32_t>(5);  // pages per device row in the gathered tensor
    constexpr uint32_t cb_in = 0, cb_zero = 1, cb_mask = 2, devices = 4, chunk_bytes = tile_rows::tile_bytes;
    constexpr auto gathered_args = TensorAccessorArgs<0>();
    constexpr auto mask_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(gathered_args, gathered_addr);
    const auto mask = TensorAccessor(mask_args, mask_addr);
    const Noc noc;

    CircularBuffer(cb_zero).reserve_back(1);
    tile_rows::fill_words(CircularBuffer(cb_zero).get_write_ptr(), chunk_bytes / 4, 0);
    CircularBuffer(cb_zero).push_back(1);
    for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {
        for (uint32_t r = 0; r < rows; ++r) {
            CircularBuffer(cb_in).reserve_back(devices);
            CircularBuffer(cb_mask).reserve_back(1);
            for (uint32_t d = 0; d < devices; ++d) {
                noc.async_read(  // device d's row r, chunk c = one 2 KB page of the gathered pages
                    gathered,
                    CircularBuffer(cb_in),
                    chunk_bytes,
                    {.page_id = (d * rows + r) * total_chunks + chunk, .offset_bytes = 0},
                    {.offset_bytes = d * chunk_bytes});
            }
            noc.async_read(
                mask, CircularBuffer(cb_mask), chunk_bytes, {.page_id = r, .offset_bytes = chunk * chunk_bytes}, {});
            noc.async_read_barrier();
            CircularBuffer(cb_in).push_back(devices);
            CircularBuffer(cb_mask).push_back(1);
        }
    }
}
