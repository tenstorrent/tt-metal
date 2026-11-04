// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for GegluDecode. Work unit = output tile (e, r, c), c < cols (the GLU half in tiles): pushes the up tile
// (page (e*rt + r) * cols2 + c) and the gate tile (same row, column c + cols) into cb_in. Units are numbered
// row-major over the output tile grid, so unit id -> (row tile, c) is id / cols, id % cols.
//
// compile-time args: [cb_in, cols2 (input tiles per row), cols (output tiles per row)] + TensorAccessorArgs(x)
// runtime args: [x_addr, start unit, unit count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cols2 = get_compile_time_arg_val(1);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    const uint32_t page = get_local_cb_interface(cb_in).fifo_page_size;
    const auto s_x = TensorAccessor(x_args, x_addr, page);
    Noc noc;
    CircularBuffer cbi(cb_in);
    for (uint32_t id = start; id < start + count; ++id) {
        const uint32_t row = id / cols;
        const uint32_t c = id - row * cols;
        const uint32_t p = row * cols2 + c;
        cbi.reserve_back(2);
        noc.async_read(s_x, cbi, page, {.page_id = p}, {.offset_bytes = 0});
        noc.async_read(s_x, cbi, page, {.page_id = p + cols}, {.offset_bytes = page});
        noc.async_read_barrier();
        cbi.push_back(2);
    }
}
