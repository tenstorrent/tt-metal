// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation loader + output writer (BRISC, NOC1) for the GLM-5.3-Flash expert down stream
// (custom_ops/expert_stream/op.py).
//
// 1. Scans the routing matrix and tells the compute threads the active-expert count (mailboxes).
// 2. For every active expert e loads its SwiGLU activation tiles act[e] ([32, I] = seg_tiles 32x32 BF16 tiles of
//    the act tensor [1, E, 32, I]) into the in0 CB: compute multiplies them against the expert's down columns and
//    keeps the running sum over the experts in DST.
// 3. Writes the `cols` accumulated [32, 32] output tiles of this core (output tile columns col0 ..) into
//    out [1, 1, 32, H] (TILE, interleaved: tile page = column). With no active expert the columns are zeroed.
//
// runtime args: [routing_addr, act_addr, out_addr, col0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "routing_scan.hpp"

void kernel_main() {
    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_r = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(4);
    constexpr uint32_t M = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr auto routing_args = TensorAccessorArgs<7>();
    constexpr auto act_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t tile_page = 2048;

    const auto s_r = TensorAccessor(routing_args, routing_addr, tile_page);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_page);

    cb_reserve_back(cb_r, E_TILES);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    for (uint32_t i = 0; i < E_TILES; ++i) {
        noc_async_read(s_r.get_noc_addr(i), r_l1 + i * tile_page, tile_page);
    }
    noc_async_read_barrier();
    uint16_t active[E];
    const uint32_t n = scan_active<E, M>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1), active);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_a, seg_tiles);
        const uint32_t a_l1 = get_write_ptr(cb_a);
        const uint32_t page0 = static_cast<uint32_t>(active[i]) * seg_tiles;
        for (uint32_t k = 0; k < seg_tiles; ++k) {
            noc_async_read(s_act.get_noc_addr(page0 + k), a_l1 + k * tile_page, tile_page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, seg_tiles);
    }

    if (n == 0) {  // no active expert: this core's output columns are zero
        cb_reserve_back(cb_out, 1);
        const uint32_t z = get_write_ptr(cb_out);
        volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t j = 0; j < tile_page / 4; ++j) {
            z32[j] = 0;
        }
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(z, s_out.get_noc_addr(col0 + c), tile_page);
        }
        noc_async_write_barrier();
        return;
    }
    cb_wait_front(cb_out, cols);
    const uint32_t o_l1 = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        noc_async_write(o_l1 + c * tile_page, s_out.get_noc_addr(col0 + c), tile_page);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
}
