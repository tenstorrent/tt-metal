// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill output, reader (NCRISC). Item = (head h, 32-token block tt): the scan output's 4 fp32 tiles
// (h, tt, 0..3) of [H, T, 128] and z's 4 bf16 tiles (tt, 4h .. 4h+3) of [1, T, H * 128]; constants once.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: o, z, consts, Nitems, ncores, grid_x, Tt, Wt (z tensor width in tiles), Zoff (z's first tile column)
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(5), 1, first, n);
    const uint32_t Tt = get_common_arg_val<uint32_t>(6), Wt = get_common_arg_val<uint32_t>(7);
    const uint32_t Zoff = get_common_arg_val<uint32_t>(8);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto z_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto zs = TensorAccessor(z_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(2), 4096);
    constexpr uint32_t cb_o = 0, cb_z = 1, cb_c = 2, NCONST = 5;  // consts: w row tiles 0..3, ONES

    cb_reserve_back(cb_c, NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc_async_read(cs.get_noc_addr(i), get_write_ptr(cb_c) + i * 4096, 4096);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, NCONST);
    for (uint32_t it = first; it < first + n; ++it) {
        const uint32_t h = it / Tt, tt = it % Tt;
        cb_reserve_back(cb_o, 4);
        cb_reserve_back(cb_z, 4);
        for (uint32_t d = 0; d < 4; ++d) {
            noc_async_read(os.get_noc_addr((h * Tt + tt) * 4 + d), get_write_ptr(cb_o) + d * 4096, 4096);
            noc_async_read(zs.get_noc_addr(tt * Wt + Zoff + h * 4 + d), get_write_ptr(cb_z) + d * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_o, 4);
        cb_push_back(cb_z, 4);
    }
}
