// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused q-prep (q_nope @ w_uk + rope(q_pe) + head concat, row-major output): reader.
// q [1,1,1,H*256] TILE (page 8h + t: nope tiles t < 6, pe tiles 6, 7), wuk [1,H,192,512] TILE (page (h*6 + k)*16 + n),
// cos/sin [1,1,1,64] TILE (2 pages), tm [1,1,32,32] TILE. Core roles (RT 0): 0 = absorb core (h, j): output tiles n =
// 2j, 2j+1 of head h; 1 = rope core of head h. CT: 0 cb_x, 1 cb_w, 2 cb_c, 3 cb_s, 4 cb_t, 5 S (query rows, <= 8), then
// TensorAccessorArgs(q), (wuk), (cos), (sin), (tm).  Common: addrs q, wuk, cos, sin, tm.  RT: role, h, j.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_t = get_compile_time_arg_val(4);
    constexpr uint32_t TB = 2048;
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr uint32_t RB = ((32 * S + 63) / 64) * 64;
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t role = get_arg_val<uint32_t>(0), h = get_arg_val<uint32_t>(1), j = get_arg_val<uint32_t>(2);
    const auto qa = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0), TB);
    CircularBuffer x(cb_x);
    if (role == 0) {
        const auto wa = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), TB);
        CircularBuffer w(cb_w);
        x.reserve_back(6);
        w.reserve_back(12);
        uint32_t px = x.get_write_ptr(), pw = w.get_write_ptr();
        for (uint32_t k = 0; k < 6; ++k) {
            noc_async_read(qa.get_noc_addr(8 * h + k, 0), px + k * TB, RB);
            noc_async_read(qa.get_noc_addr(8 * h + k, 512), px + k * TB + 512, RB);
            for (uint32_t n = 0; n < 2; ++n) {
                noc_async_read(wa.get_noc_addr((h * 6 + k) * 16 + 2 * j + n), pw + (n * 6 + k) * TB, TB);
            }
        }
        noc_async_read_barrier();
        w.push_back(12);
        x.push_back(6);
    } else {
        const auto ca = TensorAccessor(c_args, get_common_arg_val<uint32_t>(2), TB);
        const auto sa = TensorAccessor(s_args, get_common_arg_val<uint32_t>(3), TB);
        const auto ta = TensorAccessor(t_args, get_common_arg_val<uint32_t>(4), TB);
        CircularBuffer c(cb_c), s(cb_s), t(cb_t);
        x.reserve_back(2);
        c.reserve_back(2);
        s.reserve_back(2);
        t.reserve_back(1);
        uint32_t px = x.get_write_ptr(), pc = c.get_write_ptr(), ps = s.get_write_ptr(), pt = t.get_write_ptr();
        noc_async_read(ta.get_noc_addr(0), pt, TB);
        for (uint32_t k = 0; k < 2; ++k) {
            noc_async_read(qa.get_noc_addr(8 * h + 6 + k, 0), px + k * TB, RB);
            noc_async_read(qa.get_noc_addr(8 * h + 6 + k, 512), px + k * TB + 512, RB);
            noc_async_read(ca.get_noc_addr(k, 0), pc + k * TB, RB);
            noc_async_read(ca.get_noc_addr(k, 512), pc + k * TB + 512, RB);
            noc_async_read(sa.get_noc_addr(k, 0), ps + k * TB, RB);
            noc_async_read(sa.get_noc_addr(k, 512), ps + k * TB + 512, RB);
        }
        noc_async_read_barrier();
        t.push_back(1);
        s.push_back(2);
        c.push_back(2);
        x.push_back(2);
    }
}
