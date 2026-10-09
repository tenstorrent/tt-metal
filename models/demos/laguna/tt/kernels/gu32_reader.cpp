// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched (32-token) decode routed gate/up reader over column-page weights (Laguna). Core (n, g) owns gate/up output
// column n for the active experts a with a % G == g. Pushes: cb_meta <- unit count; cb_x <- the Kt activation tiles
// (once); per unit cb_wv <- the expert's routing-weight tile, cb_w <- the gate column then the up column, CHUNK tiles
// per push (each chunk one contiguous read from its column page).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t x_page = get_compile_time_arg_val(4);
    constexpr uint32_t w_tile = get_compile_time_arg_val(5);
    constexpr uint32_t sp_page = get_compile_time_arg_val(6);
    constexpr uint32_t grid_x = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_wv = 2, cb_meta = 3, cb_sp = 4;
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wv_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<wv_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t wv_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t n = core / G, g = core % G;
    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const auto wv = TensorAccessor(wv_args, wv_addr, x_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t units = 0, seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] != 0) {
            units += (seen % G == g) ? 1 : 0;
            ++seen;
        }
    }
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = units;
    cb_push_back(cb_meta, 1);
    if (units == 0) {
        return;
    }
    cb_reserve_back(cb_x, Kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < Kt; ++k) {
        noc_async_read_tile(k, x, x_l1 + k * x_page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);

    seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] == 0) {
            continue;
        }
        const bool mine = (seen % G == g);
        ++seen;
        if (!mine) {
            continue;
        }
        cb_reserve_back(cb_wv, 1);
        noc_async_read_tile(e, wv, get_write_ptr(cb_wv));
        noc_async_read_barrier();
        cb_push_back(cb_wv, 1);
        for (uint32_t half = 0; half < 2; ++half) {
            // tile (k = 0) of the column; its Kt tiles follow contiguously in the column's shard
            const uint64_t col = w.get_noc_addr(e * Kt * 2 * Nt + half * Nt + n);
            for (uint32_t k0 = 0; k0 < Kt; k0 += chunk) {
                cb_reserve_back(cb_w, chunk);
                noc_async_read(col + k0 * w_tile, get_write_ptr(cb_w), chunk * w_tile);
                noc_async_read_barrier();
                cb_push_back(cb_w, chunk);
            }
        }
    }
}
