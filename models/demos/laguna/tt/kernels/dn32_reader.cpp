// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode down + expert-sum reader over column-page weights (Laguna). Core c owns output columns
// c*CPC .. c*CPC+CPC-1. Per active expert pushes its Kt activation tiles (the [1, E, 32, I] SwiGLU output row) and
// its CPC down column pages (CPC*Kt tiles, one contiguous read per column).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nh = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t x_page = get_compile_time_arg_val(3);
    constexpr uint32_t w_tile = get_compile_time_arg_val(4);
    constexpr uint32_t sp_page = get_compile_time_arg_val(5);
    constexpr uint32_t grid_x = get_compile_time_arg_val(6);
    constexpr uint32_t CPC = get_compile_time_arg_val(7);
    constexpr uint32_t EG = get_compile_time_arg_val(8);  // expert groups: active expert a goes to group a % EG
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_sp = 3;
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t x_rows = get_common_arg_val<uint32_t>(4);  // rows of the activation tiles that are read
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t c0 = (core / EG) * CPC, eg = core % EG;
    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);
    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    // sp_rows > 1: per-token routing rows (router32.route_local_rows); an expert is active if any row routes to it
    const uint32_t sp_rows = get_common_arg_val<uint32_t>(3);
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
    uint32_t na = 0, seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] != 0) {
            na += (seen % EG == eg) ? 1 : 0;
            ++seen;
        }
    }
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = na;
    cb_push_back(cb_meta, 1);
    seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] == 0) {
            continue;
        }
        const bool mine = (seen % EG == eg);
        ++seen;
        if (!mine) {
            continue;
        }
        cb_reserve_back(cb_w, CPC * Kt);
        const uint32_t w_l1 = get_write_ptr(cb_w);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_read(w.get_noc_addr(e * Kt * Nh + c0 + j), w_l1 + j * Kt * w_tile, Kt * w_tile);
        }
        cb_reserve_back(cb_x, Kt);
        const uint32_t x_l1 = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < Kt; ++k) {
            if (x_rows < 16) {
                // only rows 0..x_rows-1 are real: their 32-byte rows of face 0 and face 1 (other rows keep stale
                // data; matmul rows are independent, so they only reach output rows the caller never reads)
                const uint64_t src = x.get_noc_addr(e * Kt + k);
                noc_async_read(src, x_l1 + k * x_page, x_rows * 32);
                noc_async_read(src + 512, x_l1 + k * x_page + 512, x_rows * 32);
            } else {
                noc_async_read_tile(e * Kt + k, x, x_l1 + k * x_page);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, Kt);
        cb_push_back(cb_w, CPC * Kt);
    }
}
