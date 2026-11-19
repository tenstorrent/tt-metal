// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for GegluPrefill. Work unit = B consecutive output tiles (row r, columns c0 .. c0 + B - 1) of the
// [rows, cols] output tile grid, numbered row-major: unit id -> (r, c0) = (id / (cols / B), (id % (cols / B)) * B).
// Pushes the B up tiles (input page r * cols2 + c0 + i) then the B gate tiles (same row, column + cols) into cb_in,
// one read barrier per unit.
//
// compile-time args: [cb_in, cols2 (input tiles per row), cols (output tiles per row), B] + TensorAccessorArgs(x)
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
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_row = cols / B;
    constexpr auto x_args = TensorAccessorArgs<4>();
    const uint32_t page = get_local_cb_interface(cb_in).fifo_page_size;
    const auto s_x = TensorAccessor(x_args, x_addr, page);
    Noc noc;
    CircularBuffer cbi(cb_in);
    for (uint32_t id = start; id < start + count; ++id) {
        const uint32_t row = id / blocks_per_row;
        const uint32_t c0 = (id - row * blocks_per_row) * B;
        const uint32_t p = row * cols2 + c0;
        cbi.reserve_back(2 * B);
        for (uint32_t i = 0; i < B; ++i) {
            noc.async_read(s_x, cbi, page, {.page_id = p + i}, {.offset_bytes = i * page});
        }
        for (uint32_t i = 0; i < B; ++i) {
            noc.async_read(s_x, cbi, page, {.page_id = p + cols + i}, {.offset_bytes = (B + i) * page});
        }
        noc.async_read_barrier();
        cbi.push_back(2 * B);
    }
}
