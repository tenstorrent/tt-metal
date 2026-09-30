// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the V4.1 query head layout (heads_q_compute.cpp): q is the tiled [.., S, H * D] projection (heads side by
// side, D = (NT + RT) tiles, the rope tail last). Per unit (tile row r, head h): the head's NT no-rope tiles to
// cb_nope and RT tail tiles to cb_tail; at every new tile row the RT cos and sin tiles of that row; once, the
// rotation tile.
//
// compile_time_args = [NT, RT, H, TensorAccessorArgs(q), (cos), (sin), (trans)]
// runtime args      = [q_addr, cos_addr, sin_addr, trans_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t trans_addr = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);
    const uint32_t unit_count = get_arg_val<uint32_t>(5);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t DT = NT + RT;
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto cos_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_nope = 0, cb_tail = 1, cb_cos = 2, cb_sin = 3, cb_trans = 4;
    if (unit_count == 0) {
        return;
    }

    const auto q = TensorAccessor(q_args, q_addr);
    const auto cos_t = TensorAccessor(cos_args, cos_addr);
    const auto sin_t = TensorAccessor(sin_args, sin_addr);
    const auto trans_t = TensorAccessor(trans_args, trans_addr);
    Noc noc;
    DataflowBuffer nope(cb_nope);
    DataflowBuffer tail(cb_tail);
    DataflowBuffer cos(cb_cos);
    DataflowBuffer sin(cb_sin);
    DataflowBuffer trans(cb_trans);
    const uint32_t page = get_local_cb_interface(cb_nope).fifo_page_size;  // bf16 tile

    trans.reserve_back(1);
    noc.async_read(trans_t, trans, page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    trans.push_back(1);

    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / H;
        const uint32_t h = u - r * H;
        if (r != row) {
            row = r;
            cos.reserve_back(RT);
            sin.reserve_back(RT);
            for (uint32_t t = 0; t < RT; ++t) {
                noc.async_read(cos_t, cos, page, {.page_id = r * RT + t}, {.offset_bytes = t * page});
                noc.async_read(sin_t, sin, page, {.page_id = r * RT + t}, {.offset_bytes = t * page});
            }
            noc.async_read_barrier();
            cos.push_back(RT);
            sin.push_back(RT);
        }
        const uint32_t first = (r * H + h) * DT;
        tail.reserve_back(RT);
        nope.reserve_back(NT);
        for (uint32_t t = 0; t < RT; ++t) {
            noc.async_read(q, tail, page, {.page_id = first + NT + t}, {.offset_bytes = t * page});
        }
        for (uint32_t t = 0; t < NT; ++t) {
            noc.async_read(q, nope, page, {.page_id = first + t}, {.offset_bytes = t * page});
        }
        noc.async_read_barrier();
        tail.push_back(RT);
        nope.push_back(NT);
    }
}
