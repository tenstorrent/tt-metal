// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Generates the reduce scaler (1.0) and the ones-column tile once, then writes each item's output tile to page
// r * slices + s of the [rows, slices] tile grid.
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_ones = 3;
constexpr uint32_t cb_out = 6;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t slices = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr auto out_args = TensorAccessorArgs<4>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t out_tile = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile);

    constexpr uint16_t one_bf16 = 0x3F80U;
    generate_tile_with_bfloat16_value(cb_scaler, one_bf16);  // the reduce scaler
    generate_matmul_row_reduce_tile(cb_ones);                // ones in column 0: the matmul row sum

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(work, out_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
