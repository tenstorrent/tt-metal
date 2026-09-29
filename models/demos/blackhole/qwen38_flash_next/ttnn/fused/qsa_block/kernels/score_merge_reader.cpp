// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge reader, one core per 1024-column chunk: the zero tile, then per row the four devices' 2 KB chunks
// of the gathered score rows and the mask chunk.  CBs: 0 device chunks (bf16, 4), 1 zero (bf16, 1), 2 mask (bf16, 1).
// Compile-time args: TensorAccessorArgs gathered ([1, 1, 4 * rows, W] ROW_MAJOR), mask ([1, 1, rows, W]).
// Runtime args: 0 gathered, 1 mask addresses, 2 rows, 3 first chunk, 4 chunks on this core (one after another),
// 5 total chunks (the gathered tensor holds the rows as 1024-column pages: device d row r chunk c at page (d * rows +
// r) * total + c).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t first_chunk = get_arg_val<uint32_t>(3);
    const uint32_t chunks = get_arg_val<uint32_t>(4);
    const uint32_t total_chunks = get_arg_val<uint32_t>(5);  // pages per device row in the gathered tensor
    constexpr uint32_t CB_IN = 0, CB_ZERO = 1, CB_MASK = 2, DEVICES = 4, CHUNK_BYTES = tile_rows::TILE_BYTES;
    constexpr auto gathered_args = TensorAccessorArgs<0>();
    constexpr auto mask_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(gathered_args, gathered_addr);
    const auto mask = TensorAccessor(mask_args, mask_addr);

    cb_reserve_back(CB_ZERO, 1);
    tile_rows::fill_words(get_write_ptr(CB_ZERO), CHUNK_BYTES / 4, 0);
    cb_push_back(CB_ZERO, 1);
    for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {
        for (uint32_t r = 0; r < rows; ++r) {
            FUSED_ZONE("fz_qs_sm_r_row");
            cb_reserve_back(CB_IN, DEVICES);
            cb_reserve_back(CB_MASK, 1);
            const uint32_t in_l1 = get_write_ptr(CB_IN);
            for (uint32_t d = 0; d < DEVICES; ++d) {
                noc_async_read(  // device d's row r, chunk c = one 2 KB page of the gathered pages
                    gathered.get_noc_addr((d * rows + r) * total_chunks + chunk, 0),
                    in_l1 + d * CHUNK_BYTES,
                    CHUNK_BYTES);
            }
            noc_async_read(mask.get_noc_addr(r, chunk * CHUNK_BYTES), get_write_ptr(CB_MASK), CHUNK_BYTES);
            noc_async_read_barrier();
            cb_push_back(CB_IN, DEVICES);
            cb_push_back(CB_MASK, 1);
        }
    }
}
