// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode gate/up writer (Laguna): same unit rule as the reader; writes unit (e, n) to tile e*Nt + n of the
// [1, E, 32, I] output (inactive experts' tiles unwritten).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sp2 = 7, cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t n = core / G, g = core % G;
    const auto out = TensorAccessor(out_args, out_addr, page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);
    const uint32_t sp_l1 = get_write_ptr(cb_sp2);
    // sp_rows > 1: per-token routing rows (router32.route_local_rows); an expert is active if any row routes to it
    const uint32_t sp_rows = get_common_arg_val<uint32_t>(2);
    for (uint32_t r = 0; r < sp_rows; ++r) {
        noc_async_read(sp.get_noc_addr(r), sp_l1 + r * sp_page, sp_page);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    for (uint32_t r = 1; r < sp_rows; ++r) {
        for (uint32_t e = 0; e < E; ++e) {
            spv[e] |= spv[r * (sp_page / 2) + e];
        }
    }
    uint32_t seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] == 0) {
            continue;
        }
        const bool mine = (seen % G == g);
        ++seen;
        if (!mine) {
            continue;
        }
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(e * Nt + n, out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
