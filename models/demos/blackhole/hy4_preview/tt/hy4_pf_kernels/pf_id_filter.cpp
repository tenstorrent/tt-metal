// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// pool-prefill: global top-k ids of R query rows (natural token positions, 0xFFFFFFFF = invalid) -> this column's LOCAL
// pool rows, sentinels as a contiguous tail (row loop version of hy4_pool id_filter, no owner replication). Token t
// lives on column (t >> 6) & 7 at local row ((t >> 9) << 6) | (t & 63).
//   gids uint32 RM [R,1,1,K]   colinfo uint32 page of 64 B (element 0 = column, element 1 = number of valid tokens: ids
//   >= it are dropped)   out uint32 RM [1,1,R,K]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr auto g_args = TensorAccessorArgs<4>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = K * 4;
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_g = get_common_arg_val<uint32_t>(0), a_c = get_common_arg_val<uint32_t>(1),
                   a_o = get_common_arg_val<uint32_t>(2);
    Noc noc;
    const auto g_acc = TensorAccessor(g_args, a_g, RB);
    const auto c_acc = TensorAccessor(c_args, a_c, 64);
    const auto o_acc = TensorAccessor(o_args, a_o, RB);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + RB);
    volatile tt_l1_ptr uint32_t* cid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 2 * RB);
    noc.async_read(c_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 2 * RB});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t c = cid[0], nvalid = cid[1];
    for (uint32_t s = core; s < R; s += NC) {
        noc.async_read(g_acc, cb, RB, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();
        uint32_t n = 0;
        for (uint32_t i = 0; i < K; ++i) {
            const uint32_t t = in[i];
            if (t < nvalid && ((t >> 6) & 7u) == c) {
                out[n++] = ((t >> 9) << 6) | (t & 63u);
            }
        }
        for (uint32_t i = n; i < K; ++i) {
            out[i] = 0xFFFFFFFFu;
        }
        noc.async_write(cb, o_acc, RB, {.offset_bytes = RB}, {.page_id = s, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
