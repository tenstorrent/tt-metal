// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused attention back (value up-projection + head concat + gate): reader. Core i = (head h = i / NT, output
// tile n = i % NT). x  [1,H,1,KT*32] TILE (head-major o_lat, tile page h*KT + k), w [1,H,KT*32,NT*32] TILE (page (h*KT
// + k)*NT + n), g [1,1,1,H*NT*32] TILE (page h*NT + n) CT: 0 cb_x, 1 cb_w, 2 cb_g, 3 KT, 4 NT, 5 S (query rows, <= 8),
// then TensorAccessorArgs(x), (w), (g).  Common: x, w, g addr.  RT: i.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t TB = 2048;
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr uint32_t RB = ((32 * S + 63) / 64) * 64;  // bytes of rows 0..S-1 of one face, 64 B aligned
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t h = i / NT, n = i % NT;
    const auto xa = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), TB);
    const auto wa = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), TB);
    const auto ga = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), TB);
    CircularBuffer x(cb_x), w(cb_w), g(cb_g);
    x.reserve_back(KT);
    w.reserve_back(KT);
    g.reserve_back(1);
    uint32_t px = x.get_write_ptr(), pw = w.get_write_ptr(), pg = g.get_write_ptr();
    noc_async_read(ga.get_noc_addr(h * NT + n, 0), pg, RB);
    noc_async_read(ga.get_noc_addr(h * NT + n, 512), pg + 512, RB);
    for (uint32_t k = 0; k < KT; ++k) {
        noc_async_read(wa.get_noc_addr((h * KT + k) * NT + n), pw + k * TB, TB);
        // only row 0 of the head-major x tiles is real: rows 0..1 of face 0 (cols 0-15) and face 1 (cols 16-31), 64 B
        // each (DRAM reads are 64 B aligned); the rest of the CB tile is never read into row 0 of the product
        noc_async_read(xa.get_noc_addr(h * KT + k, 0), px + k * TB, RB);
        noc_async_read(xa.get_noc_addr(h * KT + k, 512), px + k * TB + 512, RB);
    }
    noc_async_read_barrier();
    g.push_back(1);
    w.push_back(KT);
    x.push_back(KT);
}
