// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for RopeDecode. Work unit = (row tile rt, column tile c < half_t) of x's padded tile grid: pushes the two
// x tiles of the rotate_half pair (c, c + half_t) into cb_x, and the user's cos / sin tiles of the same two columns
// into cb_cos / cb_sin (the user's [1, D] row sits in row 0 of each tile; compute row-broadcasts it).
//
// compile-time args: [cb_x, cb_cos, cb_sin, ht (row tiles per user), cols_t, half_t] + TensorAccessorArgs(x, cos, sin)
// runtime args: [x_addr, cos_addr, sin_addr, start unit, unit count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t ht = get_compile_time_arg_val(3);
    constexpr uint32_t cols_t = get_compile_time_arg_val(4);
    constexpr uint32_t half_t = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const uint32_t page = get_local_cb_interface(cb_x).fifo_page_size;  // one bf16 tile
    const auto s_x = TensorAccessor(x_args, x_addr, page);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, page);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, page);
    Noc noc;
    CircularBuffer cbx(cb_x), cbc(cb_cos), cbs(cb_sin);

    for (uint32_t id = start; id < start + count; ++id) {
        const uint32_t rt = id / half_t;
        const uint32_t c = id - rt * half_t;
        const uint32_t u = rt / ht;  // the user of this row tile
        const uint32_t xp = rt * cols_t + c;
        const uint32_t tp = u * cols_t + c;
        cbx.reserve_back(2);
        cbc.reserve_back(2);
        cbs.reserve_back(2);
        noc.async_read(s_x, cbx, page, {.page_id = xp}, {.offset_bytes = 0});
        noc.async_read(s_x, cbx, page, {.page_id = xp + half_t}, {.offset_bytes = page});
        noc.async_read(s_cos, cbc, page, {.page_id = tp}, {.offset_bytes = 0});
        noc.async_read(s_cos, cbc, page, {.page_id = tp + half_t}, {.offset_bytes = page});
        noc.async_read(s_sin, cbs, page, {.page_id = tp}, {.offset_bytes = 0});
        noc.async_read(s_sin, cbs, page, {.page_id = tp + half_t}, {.offset_bytes = page});
        noc.async_read_barrier();
        cbx.push_back(2);
        cbc.push_back(2);
        cbs.push_back(2);
    }
}
