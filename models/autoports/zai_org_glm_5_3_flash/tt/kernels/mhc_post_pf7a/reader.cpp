// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7a prefill post-mix — reader (common.hpp; Stage 16's mhc_post_pf_reader with the coefficient tiles packed for
// the one-pass compute). Compile-time: CT (column tiles per stream), Y_PAGE (bytes of a y tile), BATCH. Runtime: y, x
// (flat), post, comb addresses, first unit, unit count. Unit u = (row tile u / CT, column tile u % CT).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "common.hpp"

using namespace mpf7;

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t Y_PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t BATCH = get_compile_time_arg_val(2);  // units per read barrier (CBs hold 2 batches)
    static_assert(CT % BATCH == 0, "a batch must not cross a row tile");
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    if (n == 0) {
        return;
    }
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = Y_PAGE};
    const InterleavedAddrGen<true> pg{.bank_base_address = post_addr, .page_size = FP32_TILE};
    const InterleavedAddrGen<true> cg{.bank_base_address = comb_addr, .page_size = FP32_TILE};
    const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = FP32_TILE};

    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t r = u / CT, c = u % CT;
        if (r != cur_r) {  // the row tile's 8 coefficient tiles (the compute pops the previous row's first)
            cur_r = r;
            cb_reserve_back(CB_RAW, 2);
            const uint32_t raw = get_write_ptr(CB_RAW);
            noc_async_read(pg.get_noc_addr(r), raw, FP32_TILE);
            noc_async_read(cg.get_noc_addr(r), raw + FP32_TILE, FP32_TILE);
            cb_reserve_back(CB_COEF, N_COEF);
            const uint32_t co = get_write_ptr(CB_COEF);
            // zero the whole tiles first: the broadcast masks multiply every other lane by 0, and 0 * NaN (stale L1)
            // would poison the broadcast value
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t off = 0; off < N_COEF * FP32_TILE; off += MEM_ZEROS_SIZE) {
                noc_async_read(
                    zeros,
                    co + off,
                    MEM_ZEROS_SIZE < N_COEF * FP32_TILE - off ? MEM_ZEROS_SIZE : N_COEF * FP32_TILE - off);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* src_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw);
            volatile tt_l1_ptr uint32_t* src_c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw + FP32_TILE);
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                volatile tt_l1_ptr uint32_t* k_t =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(co + 2 * s * FP32_TILE);
                volatile tt_l1_ptr uint32_t* l_t =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(co + (2 * s + 1) * FP32_TILE);
                for (uint32_t row = 0; row < 32; ++row) {
                    // lane 0 of each SFPLOAD slot (left face even / odd columns, right face even / odd): columns
                    // 0 / 1 / 16 / 17 (an SFPU lane c of slot +2 is column 2c + 1, not 8 + c)
                    k_t[tile_off(row, 0)] = src_p[tile_off(row, s)];       // post_s
                    k_t[tile_off(row, 1)] = src_c[tile_off(row, s)];       // C[0][s] (comb column 4 i + s)
                    k_t[tile_off(row, 16)] = src_c[tile_off(row, 4 + s)];  // C[1][s]
                    k_t[tile_off(row, 17)] = src_c[tile_off(row, 8 + s)];  // C[2][s]
                    l_t[tile_off(row, 0)] = src_c[tile_off(row, 12 + s)];  // C[3][s]
                }
            }
            cb_push_back(CB_COEF, N_COEF);  // CB_RAW is reader-private scratch: never pushed
        }
        // BATCH units per barrier: the host gives every core a BATCH-aligned multiple of BATCH units and CT is a
        // multiple of BATCH, so a batch never crosses a row tile and every push is exactly BATCH units (the CBs hold 2
        // batches: push sizes must tile the CB capacity, a push across the CB end trips the watcher)
        constexpr uint32_t nb = BATCH;
        cb_reserve_back(CB_Y, nb);
        cb_reserve_back(CB_X, N_STREAMS * nb);
        const uint32_t yw = get_write_ptr(CB_Y);
        const uint32_t xw = get_write_ptr(CB_X);
        for (uint32_t j = 0; j < nb; ++j) {
            noc_async_read(yg.get_noc_addr(r * CT + c + j), yw + j * Y_PAGE, Y_PAGE);
            for (uint32_t i = 0; i < N_STREAMS; ++i) {  // stream i of the flat residual: column tile i CT + c + j
                noc_async_read(
                    xg.get_noc_addr(r * N_STREAMS * CT + i * CT + c + j),
                    xw + (j * N_STREAMS + i) * FP32_TILE,
                    FP32_TILE);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_Y, nb);
        cb_push_back(CB_X, N_STREAMS * nb);
        u += nb - 1;  // (the loop's ++u)
    }
}
