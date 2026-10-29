// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// q / k head norm + partial RoPE, reader (NCRISC): the norm weight's 8 tiles once, then per group (head h,
// row tile r) the head's 8 tiles of t ([1, heads, C, 256] head-major tiles) and the cos / sin tile of row r
// (the two rotated halves share one; [C, 64] tiles).
// Compile-time args: 0 Rt. Runtime args: 0 first group, 1 groups, 2 t, 3 weight, 4 cos, 5 sin addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_t = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, Dt = 8;
    constexpr uint32_t tile = get_tile_size(cb_t);
    const uint32_t first = get_arg_val<uint32_t>(0), groups = get_arg_val<uint32_t>(1);
    auto gen = [&](uint32_t i) {
        return InterleavedAddrGenFast<true>{
            .bank_base_address = get_arg_val<uint32_t>(i), .page_size = tile, .data_format = DataFormat::Float16_b};
    };
    const auto t = gen(2), w = gen(3), cs = gen(4), sn = gen(5);
    cb_reserve_back(cb_w, Dt);
    for (uint32_t d = 0; d < Dt; d++) {
        noc_async_read_tile(d, w, get_write_ptr(cb_w) + d * tile);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, Dt);
    for (uint32_t g = first; g < first + groups; g++) {
        const uint32_t r = g % Rt;
        cb_reserve_back(cb_t, Dt);
        cb_reserve_back(cb_cos, 1);
        cb_reserve_back(cb_sin, 1);
        for (uint32_t d = 0; d < Dt; d++) {
            noc_async_read_tile(g * Dt + d, t, get_write_ptr(cb_t) + d * tile);
        }
        noc_async_read_tile(r * 2, cs, get_write_ptr(cb_cos));
        noc_async_read_tile(r * 2, sn, get_write_ptr(cb_sin));
        noc_async_read_barrier();
        cb_push_back(cb_t, Dt);
        cb_push_back(cb_cos, 1);
        cb_push_back(cb_sin, 1);
    }
}
