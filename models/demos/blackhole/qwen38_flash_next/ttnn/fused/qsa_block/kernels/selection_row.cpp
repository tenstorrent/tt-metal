// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// selection_row (data movement only, one core per 256-column slice; the last slice also takes the 32 sentinel
// columns): sparse[i] = (((ids[i >> 2] << 2) + offsets[r][i]) & keep[i]) | fill[i] for i < 2048, (sentinel[i - 2048] &
// keep[i]) | fill[i] after, per row r.  uint32 ROW_MAJOR rows: ids [rows, 512], offsets [1, 2048] (shared by every
// row) or [>= rows, 2048] (row r for row r), sentinel [1, 32], keep / fill [>= rows, 2080], out [rows, 2080].
// CB 0 (cb_scratch): scratch (one 8 KB page).
// Compile-time args: TensorAccessorArgs ids, offsets, sentinel, keep, fill, out.
// Runtime args: 0-5 those addresses, 6 rows, 7 slice, 8 offset rows (1 or >= rows), 9 row_first, 10 row_step (this
// core's rows: row_first, row_first + row_step, ... below rows).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(1);
    const uint32_t sentinel_addr = get_arg_val<uint32_t>(2);
    const uint32_t keep_addr = get_arg_val<uint32_t>(3);
    const uint32_t fill_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t slice = get_arg_val<uint32_t>(7);
    const uint32_t offset_rows = get_arg_val<uint32_t>(8);
    const uint32_t row_first = get_arg_val<uint32_t>(9);
    const uint32_t row_step = get_arg_val<uint32_t>(10);
    constexpr uint32_t slice_columns = 256, slice_ids = slice_columns / 4, sentinel_columns = 32, last_slice = 7,
                       expanded_columns = 2048;
    constexpr uint32_t cb_scratch = 0;
    constexpr auto ids_args = TensorAccessorArgs<0>();
    constexpr auto offsets_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto sentinel_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<sentinel_args.next_compile_time_args_offset()>();
    constexpr auto fill_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<fill_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto offsets = TensorAccessor(offsets_args, offsets_addr);
    const auto sentinel = TensorAccessor(sentinel_args, sentinel_addr);
    const auto keep = TensorAccessor(keep_args, keep_addr);
    const auto fill = TensorAccessor(fill_args, fill_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;

    const uint32_t columns = slice_columns + (slice == last_slice ? sentinel_columns : 0);
    const uint32_t first_column = slice * slice_columns;
    CircularBuffer(cb_scratch).reserve_back(1);
    // cb_scratch is reserved once and never pushed: the rows' inputs and the output row at fixed offsets
    const uint32_t scratch_l1 = CircularBuffer(cb_scratch).get_write_ptr();
    const CoreLocalMem<volatile uint32_t> ids_l1(scratch_l1);
    const CoreLocalMem<volatile uint32_t> offsets_l1(scratch_l1 + 256);
    const CoreLocalMem<volatile uint32_t> sentinel_l1(scratch_l1 + 256 + 1024);
    const CoreLocalMem<volatile uint32_t> keep_l1(scratch_l1 + 256 + 1024 + 128);
    const CoreLocalMem<volatile uint32_t> fill_l1(scratch_l1 + 256 + 1024 + 128 + 1152);
    const CoreLocalMem<volatile uint32_t> out_l1(scratch_l1 + 256 + 1024 + 128 + 2 * 1152);

    if (offset_rows == 1) {
        noc.async_read(offsets, offsets_l1, slice_columns * 4, {.page_id = 0, .offset_bytes = first_column * 4}, {});
    }
    if (slice == last_slice) {
        noc.async_read(sentinel, sentinel_l1, sentinel_columns * 4, {.page_id = 0, .offset_bytes = 0}, {});
    }
    for (uint32_t r = row_first; r < rows; r += row_step) {
        if (offset_rows != 1) {
            noc.async_read(  // row r's offsets
                offsets,
                offsets_l1,
                slice_columns * 4,
                {.page_id = r, .offset_bytes = first_column * 4},
                {});
        }
        noc.async_read(ids, ids_l1, slice_ids * 4, {.page_id = r, .offset_bytes = slice * slice_ids * 4}, {});
        noc.async_read(keep, keep_l1, columns * 4, {.page_id = r, .offset_bytes = first_column * 4}, {});
        noc.async_read(fill, fill_l1, columns * 4, {.page_id = r, .offset_bytes = first_column * 4}, {});
        noc.async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < columns; ++i) {
            const uint32_t column = first_column + i;
            const uint32_t value = column < expanded_columns ? (ids_l1[i >> 2] << 2) + offsets_l1[i]
                                                             : sentinel_l1[column - expanded_columns];
            out_l1[i] = (value & keep_l1[i]) | fill_l1[i];
        }
        noc.async_write(out_l1, out, columns * 4, {}, {.page_id = r, .offset_bytes = first_column * 4});
        noc.async_write_barrier();
    }
}
