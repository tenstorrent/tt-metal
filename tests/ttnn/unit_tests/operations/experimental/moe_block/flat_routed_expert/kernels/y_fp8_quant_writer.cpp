// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FP8_Y_PLAN.md M2 writer: each row tile's 32 e4m3 rows (out CB c_16, `columns` pages) to y (row-major, a page a row),
// and its scale page (c_17, 512 B) to y_scales (a page a row tile).
//
// Compile-time args: row_tiles, columns (out CB pages a row tile), row_bytes, row_stride_bytes (in the out CB),
// scale_page_bytes, then the TensorAccessorArgs of y and of y_scales
// Runtime args: y address, y_scales address
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t scale_page_bytes = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t out_cb = 16;
    constexpr uint32_t scale_cb = 17;
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0), row_bytes);
    const auto s = TensorAccessor(s_args, get_arg_val<uint32_t>(1), scale_page_bytes);
    for (uint32_t row = 0; row < row_tiles; ++row) {
        cb_wait_front(out_cb, columns);
        cb_wait_front(scale_cb, 1);
        const uint32_t rows_l1 = get_read_ptr(out_cb);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(rows_l1 + r * row_stride_bytes, y.get_noc_addr(row * 32 + r), row_bytes);
        }
        noc_async_write(get_read_ptr(scale_cb), s.get_noc_addr(row), scale_page_bytes);
        noc_async_write_barrier();
        cb_pop_front(out_cb, columns);
        cb_pop_front(scale_cb, 1);
    }
}
