// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The y quantize test's writer: each row tile's 32 rows of value bytes (out CB c_16, a row stride of row_stride_bytes)
// to y (row-major, a page a row), and its cells page (c_17: 32 rows x 16 B) to y_cells (a page a row tile).
//
// Compile-time args: row_tiles, out_pages (out CB pages a row tile), row_bytes, row_stride_bytes, then the
// TensorAccessorArgs of y and of y_cells. Runtime args: y address, y_cells address
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t out_pages = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_stride_bytes = get_compile_time_arg_val(3);
    constexpr auto y_args = TensorAccessorArgs<4>();
    constexpr auto c_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t out_cb = 16, cells_cb = 17, cells_page = 32 * 16;
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0), row_bytes);
    const auto c = TensorAccessor(c_args, get_arg_val<uint32_t>(1), cells_page);
    for (uint32_t row = 0; row < row_tiles; ++row) {
        cb_wait_front(out_cb, out_pages);
        cb_wait_front(cells_cb, 1);
        const uint32_t rows_l1 = get_read_ptr(out_cb);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(rows_l1 + r * row_stride_bytes, y.get_noc_addr(row * 32 + r), row_bytes);
        }
        noc_async_write(get_read_ptr(cells_cb), c.get_noc_addr(row), cells_page);
        noc_async_write_barrier();
        cb_pop_front(out_cb, out_pages);
        cb_pop_front(cells_cb, 1);
    }
}
