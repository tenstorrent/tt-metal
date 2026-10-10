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
    constexpr uint32_t core_base = get_compile_time_arg_val(9);  // first core of this kernel's grid (row-major index)
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wv_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<wv_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t wv_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - core_base;
    const uint32_t n = core / G, g = core % G;
    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const auto wv = TensorAccessor(wv_args, wv_addr, x_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    // sp_rows > 1: per-token routing rows (router32.route_local_rows); an expert is active if any row routes to it.
    // The union goes to the slot after the rows, which stay intact: with wv_addr == 0 the expert's weight tile is
    // built from them (row t, column 0 = row t's weight) instead of read from wv.
    const uint32_t sp_rows = get_common_arg_val<uint32_t>(4);
    for (uint32_t r = 0; r < sp_rows; ++r) {
        noc_async_read(sp.get_noc_addr(r), sp_l1 + r * sp_page, sp_page);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* rowsv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    volatile tt_l1_ptr uint16_t* spv = rowsv;
    if (sp_rows > 1) {
        spv = rowsv + sp_rows * (sp_page / 2);
        for (uint32_t e = 0; e < E; ++e) {
            uint16_t any = 0;
            for (uint32_t r = 0; r < sp_rows; ++r) {
                any |= rowsv[r * (sp_page / 2) + e];
            }
            spv[e] = any;
        }
    }
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
    // x multicast (n_rects > 0): core 0 reads the Kt activation tiles once and multicasts them into every core's
    // cb_x (same L1 address on each core: the CB is empty at program start), then sets each core's x-ready
    // semaphore; the other cores wait for it instead of each reading all of x (96 cores x 192 KB of NoC reads cost
    // ~60 of ~190 us at 13 active experts). Rectangles: common args 6.. as (x0, y0, x1, y1, dests) in NOC coords.
    const uint32_t n_rects = get_common_arg_val<uint32_t>(5);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    if (n_rects > 0 && core == 0) {
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_tile(k, x, x_l1 + k * x_page);
        }
        noc_async_read_barrier();
        const uint32_t sem = get_semaphore(0);
        // the semaphore multicast sends this core's own semaphore value: set it first
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem)[0] = 1;
        for (uint32_t r = 0; r < n_rects; ++r) {
            const uint32_t a = 6 + r * 5;
            const uint32_t x0 = get_common_arg_val<uint32_t>(a), y0 = get_common_arg_val<uint32_t>(a + 1);
            const uint32_t x1 = get_common_arg_val<uint32_t>(a + 2), y1 = get_common_arg_val<uint32_t>(a + 3);
            const uint32_t dests = get_common_arg_val<uint32_t>(a + 4);
            noc_async_write_multicast(x_l1, get_noc_multicast_addr(x0, y0, x1, y1, x_l1), Kt * x_page, dests);
            noc_semaphore_set_multicast(sem, get_noc_multicast_addr(x0, y0, x1, y1, sem), dests);
        }
        noc_async_write_barrier();
    }
    if (units == 0) {
        return;
    }
    cb_reserve_back(cb_x, Kt);
    if (n_rects > 0) {
        if (core != 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
        }
    } else {
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_tile(k, x, x_l1 + k * x_page);
        }
        noc_async_read_barrier();
    }
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
        if (wv_addr != 0) {
            noc_async_read_tile(e, wv, get_write_ptr(cb_wv));
            noc_async_read_barrier();
        } else {
            // bf16 tile: face (t / 16) * 2 holds rows t of columns 0-15, 16 values per row
            volatile tt_l1_ptr uint32_t* tw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_wv));
            for (uint32_t i = 0; i < x_page / 4; ++i) {
                tw[i] = 0;
            }
            volatile tt_l1_ptr uint16_t* th = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_wv));
            for (uint32_t t = 0; t < sp_rows; ++t) {
                th[(t / 16) * 512 + (t % 16) * 16] = rowsv[t * (sp_page / 2) + e];
            }
        }
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
