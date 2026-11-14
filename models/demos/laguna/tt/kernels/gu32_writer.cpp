// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode gate/up writer (Laguna): same unit rule as the reader; writes unit (e, n) to tiles (e * Mt + m) * Nt + n
// of the [1, E, 32 Mt, I] output (inactive experts' tiles unwritten; skip: also the token tiles holding no row routed
// to e -- down32 reads only routed rows).

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
    constexpr uint32_t core_base = get_compile_time_arg_val(6);  // first core of this kernel's grid
    constexpr uint32_t Mt = get_compile_time_arg_val(7);
    constexpr bool skip = get_compile_time_arg_val(8) == 1 && Mt > 1;
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - core_base;
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
    // the union of the rows goes to the slot after them (the rows stay intact for the per-tile masks)
    volatile tt_l1_ptr uint16_t* rowsv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    volatile tt_l1_ptr uint16_t* spv = rowsv + sp_rows * (sp_page / 2);
    for (uint32_t e = 0; e < E; ++e) {
        uint16_t any = 0;
        for (uint32_t r = 0; r < sp_rows; ++r) {
            any |= rowsv[r * (sp_page / 2) + e];
        }
        spv[e] = any;
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
        uint32_t mask = (1u << Mt) - 1;
        if constexpr (skip) {
            if (sp_rows > 1) {
                mask = 0;
                for (uint32_t t = 0; t < sp_rows; ++t) {
                    mask |= (rowsv[t * (sp_page / 2) + e] != 0 ? 1u : 0u) << (t / 32);
                }
            }
        }
        for (uint32_t m = 0; m < Mt; ++m) {
            if (!((mask >> m) & 1)) {
                continue;
            }
            cb_wait_front(cb_out, 1);
            noc_async_write_tile((e * Mt + m) * Nt + n, out, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
