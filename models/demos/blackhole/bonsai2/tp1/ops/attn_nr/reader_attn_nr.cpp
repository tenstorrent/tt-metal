// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for attn_nr (attention decode q/k RMSNorm + RoPE in one op). One core per head: q heads read their 8 tiles
// from the [Hq, 1, D] q slice and the q norm weight, k heads from the [1, 1, Hk*D] k slice (head h = tiles h*Dt..)
// and the k norm weight; both read cos/sin tiles 0 and 1 (the 64 rotary dims of D = 256).
// Two accessor types are never mixed in one expression (a ?: over them does not compile): q and k are separate
// branches.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Rt = get_compile_time_arg_val(1);  // rotary tiles (2: 64 dims)
    constexpr auto q_a = TensorAccessorArgs<2>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto wq_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto wk_a = TensorAccessorArgs<wq_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<wk_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();

    // runtime: 0 q, 1 k, 2 wq, 3 wk, 4 cos, 5 sin, 6 role (0 q, 1 k), 7 head index within its role
    const uint32_t role = get_arg_val<uint32_t>(6);
    const uint32_t hl = get_arg_val<uint32_t>(7);
    const uint32_t tb = get_tile_size(cb_x);  // bf16

    Noc noc;
    CircularBuffer cx(cb_x), cw(cb_w), ccs(cb_cs);
    cx.reserve_back(Dt);
    cw.reserve_back(Dt);
    ccs.reserve_back(2 * Rt);
    if (role == 0) {
        const auto x = TensorAccessor(q_a, get_arg_val<uint32_t>(0), tb);
        const auto w = TensorAccessor(wq_a, get_arg_val<uint32_t>(2), tb);
        for (uint32_t t = 0; t < Dt; t++) {
            noc.async_read(x, cx, tb, {.page_id = hl * Dt + t}, {.offset_bytes = t * tb});
            noc.async_read(w, cw, tb, {.page_id = t}, {.offset_bytes = t * tb});
        }
    } else {
        const auto x = TensorAccessor(k_a, get_arg_val<uint32_t>(1), tb);
        const auto w = TensorAccessor(wk_a, get_arg_val<uint32_t>(3), tb);
        for (uint32_t t = 0; t < Dt; t++) {
            noc.async_read(x, cx, tb, {.page_id = hl * Dt + t}, {.offset_bytes = t * tb});
            noc.async_read(w, cw, tb, {.page_id = t}, {.offset_bytes = t * tb});
        }
    }
    const auto cs = TensorAccessor(c_a, get_arg_val<uint32_t>(4), tb);
    const auto sn = TensorAccessor(s_a, get_arg_val<uint32_t>(5), tb);
    for (uint32_t t = 0; t < Rt; t++) {
        noc.async_read(cs, ccs, tb, {.page_id = t}, {.offset_bytes = t * tb});            // cos tiles 0..Rt-1
        noc.async_read(sn, ccs, tb, {.page_id = t}, {.offset_bytes = (Rt + t) * tb});     // sin tiles 0..Rt-1
    }
    noc.async_read_barrier();
    cx.push_back(Dt);
    cw.push_back(Dt);
    ccs.push_back(2 * Rt);
}
