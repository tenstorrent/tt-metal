// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill (residual add +) RMSNorm + rotation (norm_rot.py), reader (NCRISC). The const tiles into c_2 once; this
// core's token row tile: HAS_ADD: a and b tile by tile into c_0 / c_1 (the compute adds them); else x straight into
// c_3, the row the compute keeps.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t Kt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t nconst = get_common_arg_val<uint32_t>(5);
    const uint32_t me = get_arg_val<uint32_t>(0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2, cb_h = 3;
    if (me >= rows) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_h);
    const auto as = TensorAccessor(a_args, a_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, get_tile_size(cb_c));
    cb_reserve_back(cb_c, nconst);
    const uint32_t lc = get_write_ptr(cb_c);
    for (uint32_t t = 0; t < nconst; ++t) {
        noc_async_read(cs.get_noc_addr(t), lc + t * get_tile_size(cb_c), get_tile_size(cb_c));
    }
#ifdef HAS_ADD
    const auto bs = TensorAccessor(b_args, b_addr, TB);
    for (uint32_t t = 0; t < Kt; ++t) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read(as.get_noc_addr(me * Kt + t), get_write_ptr(cb_a), TB);
        noc_async_read(bs.get_noc_addr(me * Kt + t), get_write_ptr(cb_b), TB);
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
        if (t == 0) {
            cb_push_back(cb_c, nconst);
        }
    }
#else
    cb_reserve_back(cb_h, Kt);
    const uint32_t lh = get_write_ptr(cb_h);
    for (uint32_t t = 0; t < Kt; ++t) {
        noc_async_read(as.get_noc_addr(me * Kt + t), lh + t * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, nconst);
    cb_push_back(cb_h, Kt);
#endif
}
