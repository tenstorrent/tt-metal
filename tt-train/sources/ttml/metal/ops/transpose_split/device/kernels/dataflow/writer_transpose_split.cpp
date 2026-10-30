// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transpose_split writer: item i < dx_items -> dx page i (row-major (t, r)); item i >= dx_items -> tail page
// i - dx_items.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 1;
constexpr uint32_t Tt = get_compile_time_arg_val(0);
constexpr uint32_t rows_t = get_compile_time_arg_val(1);
constexpr uint32_t dx_items = get_compile_time_arg_val(2);
constexpr auto dx_args = TensorAccessorArgs<3>();
constexpr auto tail_args = TensorAccessorArgs<dx_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t dx_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tail_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto dx_acc = TensorAccessor(dx_args, dx_addr, tile_bytes);
    const auto tail_acc = TensorAccessor(tail_args, tail_addr, tile_bytes);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        cb_wait_front(cb_out, 1);
        if (work < dx_items) {
            noc_async_write_page(work, dx_acc, get_read_ptr(cb_out));
        } else {
            noc_async_write_page(work - dx_items, tail_acc, get_read_ptr(cb_out));
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
