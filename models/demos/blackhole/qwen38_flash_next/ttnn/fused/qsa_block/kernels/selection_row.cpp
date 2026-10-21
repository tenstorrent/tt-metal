// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// selection_row (data movement only, one core per 256-column slice; the last slice also takes the 32 sentinel
// columns): sparse[i] = (((ids[i >> 2] << 2) + offsets[r][i]) & keep[i]) | fill[i] for i < 2048, (sentinel[i - 2048] &
// keep[i]) | fill[i] after, per row r.  uint32 ROW_MAJOR rows: ids [rows, 512], offsets [1, 2048] (one row for every
// row: the decode step) or [>= rows, 2048] (row r for row r: the lanes, row u = offsets + lane u's KV region start),
// sentinel [1, 32], keep / fill [>= rows, 2080] (rows 0 .. rows-1 read), out [rows, 2080].  CB 0: scratch (one 8 KB
// page).
// Compile-time args: TensorAccessorArgs ids, offsets, sentinel, keep, fill, out.
// Runtime args: 0-5 those addresses, 6 rows, 7 slice, 8 offset rows (1 or >= rows), 9 row_first, 10 row_step (this
// core's rows: row_first, row_first + row_step, ... below rows; the lanes spread their rows over row groups of cores).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t off_addr = get_arg_val<uint32_t>(1);
    const uint32_t sentinel_addr = get_arg_val<uint32_t>(2);
    const uint32_t keep_addr = get_arg_val<uint32_t>(3);
    const uint32_t fill_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t slice = get_arg_val<uint32_t>(7);
    const uint32_t offset_rows = get_arg_val<uint32_t>(8);
    const uint32_t row_first = get_arg_val<uint32_t>(9);
    const uint32_t row_step = get_arg_val<uint32_t>(10);
    constexpr uint32_t COLS = 256, IDS = COLS / 4, TAIL = 32, LAST_SLICE = 7, EXPANDED = 2048;
    constexpr auto ids_args = TensorAccessorArgs<0>();
    constexpr auto off_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto sentinel_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<sentinel_args.next_compile_time_args_offset()>();
    constexpr auto fill_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<fill_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto off = TensorAccessor(off_args, off_addr);
    const auto sentinel = TensorAccessor(sentinel_args, sentinel_addr);
    const auto keep = TensorAccessor(keep_args, keep_addr);
    const auto fill = TensorAccessor(fill_args, fill_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t cols = COLS + (slice == LAST_SLICE ? TAIL : 0);
    const uint32_t col0 = slice * COLS;
    cb_reserve_back(0, 1);
    const uint32_t l1 = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* ids_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* off_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256);
    volatile tt_l1_ptr uint32_t* sentinel_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256 + 1024);
    volatile tt_l1_ptr uint32_t* keep_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256 + 1024 + 128);
    volatile tt_l1_ptr uint32_t* fill_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256 + 1024 + 128 + 1152);
    volatile tt_l1_ptr uint32_t* out_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256 + 1024 + 128 + 2 * 1152);

    if (offset_rows == 1) {
        noc_async_read(off.get_noc_addr(0, col0 * 4), l1 + 256, COLS * 4);
    }
    if (slice == LAST_SLICE) {
        noc_async_read(sentinel.get_noc_addr(0, 0), l1 + 256 + 1024, TAIL * 4);
    }
    for (uint32_t r = row_first; r < rows; r += row_step) {
        if (offset_rows != 1) {
            noc_async_read(off.get_noc_addr(r, col0 * 4), l1 + 256, COLS * 4);  // row r's offsets (the lanes)
        }
        noc_async_read(ids.get_noc_addr(r, slice * IDS * 4), l1, IDS * 4);
        noc_async_read(keep.get_noc_addr(r, col0 * 4), l1 + 256 + 1024 + 128, cols * 4);
        noc_async_read(fill.get_noc_addr(r, col0 * 4), l1 + 256 + 1024 + 128 + 1152, cols * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < cols; ++i) {
            const uint32_t column = col0 + i;
            const uint32_t value =
                column < EXPANDED ? (ids_l1[i >> 2] << 2) + off_l1[i] : sentinel_l1[column - EXPANDED];
            out_l1[i] = (value & keep_l1[i]) | fill_l1[i];
        }
        noc_async_write(l1 + 256 + 1024 + 128 + 2 * 1152, out.get_noc_addr(r, col0 * 4), cols * 4);
        noc_async_write_barrier();
    }
}
