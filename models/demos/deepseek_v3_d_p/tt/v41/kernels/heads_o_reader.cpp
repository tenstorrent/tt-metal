// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the V4.1 attention-output head layout (heads_o_compute.cpp): attn is row-major [1, H, S, D], one page
// per (head, token) row, D = (NT + RT) tiles wide with the rope tail last. Per unit (tile row r, head h) the head's
// 32 rows: the no-rope part (NT * 64 bytes per row) to cb_rm_nope and the tail (RT * 64 bytes) to cb_rm_tail, each
// as a row-major block of 32 contiguous rows; at every new tile row the RT cos and sin tiles; once, the rotation.
//
// compile_time_args = [NT, RT, H, S, TensorAccessorArgs(attn), (cos), (sin), (trans)]
// runtime args      = [attn_addr, cos_addr, sin_addr, trans_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t trans_addr = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);
    const uint32_t unit_count = get_arg_val<uint32_t>(5);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr auto attn_args = TensorAccessorArgs<4>();
    constexpr auto cos_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_rm_nope = 0, cb_rm_tail = 1, cb_cos = 2, cb_sin = 3, cb_trans = 4;
    constexpr uint32_t NOPE_ROW = NT * 32 * 2, TAIL_ROW = RT * 32 * 2;  // bf16 bytes
    if (unit_count == 0) {
        return;
    }

    const auto attn = TensorAccessor(attn_args, attn_addr);
    const auto cos_t = TensorAccessor(cos_args, cos_addr);
    const auto sin_t = TensorAccessor(sin_args, sin_addr);
    const auto trans_t = TensorAccessor(trans_args, trans_addr);
    Noc noc;
    DataflowBuffer nope(cb_rm_nope);
    DataflowBuffer tail(cb_rm_tail);
    DataflowBuffer cos(cb_cos);
    DataflowBuffer sin(cb_sin);
    DataflowBuffer trans(cb_trans);
    const uint32_t page = get_local_cb_interface(cb_cos).fifo_page_size;  // bf16 tile

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
        const uint32_t first = h * S + r * 32;
        nope.reserve_back(NT);
        tail.reserve_back(RT);
        for (uint32_t i = 0; i < 32; ++i) {
            noc.async_read(attn, nope, NOPE_ROW, {.page_id = first + i}, {.offset_bytes = i * NOPE_ROW});
            noc.async_read(
                attn, tail, TAIL_ROW, {.page_id = first + i, .offset_bytes = NOPE_ROW}, {.offset_bytes = i * TAIL_ROW});
        }
        noc.async_read_barrier();
        nope.push_back(NT);
        tail.push_back(RT);
    }
}
