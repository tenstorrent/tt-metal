// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transpose_split reader: item i < dx_items -> (t = i / dxt, r = i % dxt), input page r*Tt + t;
// item i >= dx_items -> t = i - dx_items, input page rows_t*Tt + t (the tail tile row).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in = 0;
constexpr uint32_t Tt = get_compile_time_arg_val(0);
constexpr uint32_t rows_t = get_compile_time_arg_val(1);
constexpr uint32_t dx_items = get_compile_time_arg_val(2);
constexpr uint32_t dxt = get_compile_time_arg_val(3);  // dx tile columns (>= rows_t)
constexpr auto in_args = TensorAccessorArgs<4>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    const auto in_acc = TensorAccessor(in_args, in_addr, tile_bytes);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        uint32_t page;
        if (work < dx_items) {
            const uint32_t t = work / dxt;
            const uint32_t r = work - t * dxt;
            page = r * Tt + t;
        } else {
            page = rows_t * Tt + (work - dx_items);
        }
        cb_reserve_back(cb_in, 1);
        noc_async_read_page(page, in_acc, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
