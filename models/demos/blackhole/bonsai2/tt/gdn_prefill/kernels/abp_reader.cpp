// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill gates (forward.py GateTiles), reader: the const tiles (fp32: dt_bias / A_neg rows; bf16: the
// 32 selection tiles) and this core's row tile of ab (3 tiles).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ab_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t rows = get_common_arg_val<uint32_t>(3);
    const uint32_t me = get_arg_val<uint32_t>(0);
    constexpr auto ab_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<ab_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_ab = 0, cb_c = 1, cb_sel = 2;
    if (me >= rows) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_ab), CT = get_tile_size(cb_c), ST = get_tile_size(cb_sel);
    const auto abs = TensorAccessor(ab_args, ab_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, CT);
    const auto ss = TensorAccessor(s_args, s_addr, ST);
    cb_reserve_back(cb_c, 4);
    cb_reserve_back(cb_sel, 32);
    cb_reserve_back(cb_ab, 3);
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(cs.get_noc_addr(t), get_write_ptr(cb_c) + t * CT, CT);
    }
    for (uint32_t t = 0; t < 32; ++t) {
        noc_async_read(ss.get_noc_addr(t), get_write_ptr(cb_sel) + t * ST, ST);
    }
    for (uint32_t t = 0; t < 3; ++t) {
        noc_async_read(abs.get_noc_addr(me * 3 + t), get_write_ptr(cb_ab) + t * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, 4);
    cb_push_back(cb_sel, 32);
    cb_push_back(cb_ab, 3);
}
