// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 2 (one core per row; more rows than cores: core c takes rows c, c + n_cores, ...): reduce the
// units' packed keys of the row (partials [W, 32], S units per tile row) and write this chip's candidate as [key,
// global column, 0 ...] (key = order-preserving bf16 value, global column = this chip's first vocabulary id (the
// uint32 at off) + local column): out [R, 8] uint32, page = row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t grid_x = get_compile_time_arg_val(3);
    constexpr uint32_t n_cores = get_compile_time_arg_val(4);
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto off_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    // per row: read the units of its tile row (S pages of 32 words), write its 8-word page [key, global column, 0 ...]
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto p = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 128);
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), 32);
    const auto off = TensorAccessor(off_args, get_common_arg_val<uint32_t>(2), 32);
    const uint32_t buf = get_write_ptr(0);
    const uint32_t offl = buf + S * 128, outl = offl + 64;
    noc_async_read(off.get_noc_addr(0), offl, 32);
    for (uint32_t row = core; row < R; row += n_cores) {
        const uint32_t tr = row / 32, r = row % 32;
        for (uint32_t s = 0; s < S; ++s) {
            noc_async_read(p.get_noc_addr(tr * S + s), buf + s * 128, 128);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t* parts = reinterpret_cast<const uint32_t*>(buf);
        uint32_t b = 0;
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t x = parts[s * 32 + r];
            b = x > b ? x : b;
        }
        volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(outl);
        res[0] = b >> 16;
        res[1] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offl)[0] + (0xFFFFu - (b & 0xFFFFu));
        for (uint32_t i = 2; i < 8; ++i) {
            res[i] = 0;
        }
        noc_async_write(outl, o.get_noc_addr(row), 32);
        noc_async_write_barrier();
    }
}
