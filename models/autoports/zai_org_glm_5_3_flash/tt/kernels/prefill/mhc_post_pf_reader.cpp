// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 prefill mHC mixes — reader (mhc_post_pf_common.hpp). Compile-time: CT (column tiles per stream), Y_PAGE
// (bytes of a y tile). Runtime: y, x (flat), coefficient A (post / pre), coefficient B (comb) addresses, first unit,
// unit count (MPP_PRE: y and comb unused).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_pf_common.hpp"

using namespace mpp;

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t Y_PAGE = get_compile_time_arg_val(1);
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
        if (r != cur_r) {  // the row tile's 20 column-0 coefficient tiles (the compute pops the previous row's first)
            cur_r = r;
            cb_reserve_back(CB_RAW, 2);
            const uint32_t raw = get_write_ptr(CB_RAW);
            noc_async_read(pg.get_noc_addr(r), raw, FP32_TILE);
#ifndef MPP_PRE
            noc_async_read(cg.get_noc_addr(r), raw + FP32_TILE, FP32_TILE);
#endif
            noc_async_read_barrier();
            cb_reserve_back(CB_COEF, N_COEF);
            const uint32_t co = get_write_ptr(CB_COEF);
            // zero the whole tiles first: the column broadcast must never see stale L1 (NaN bits) in columns 1..31
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t off = 0; off < N_COEF * FP32_TILE; off += MEM_ZEROS_SIZE) {
                noc_async_read(
                    zeros,
                    co + off,
                    MEM_ZEROS_SIZE < N_COEF * FP32_TILE - off ? MEM_ZEROS_SIZE : N_COEF * FP32_TILE - off);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr float* src_p = reinterpret_cast<volatile tt_l1_ptr float*>(raw);
            volatile tt_l1_ptr float* src_c = reinterpret_cast<volatile tt_l1_ptr float*>(raw + FP32_TILE);
            for (uint32_t k = 0; k < N_COEF; ++k) {
                volatile tt_l1_ptr float* dst = reinterpret_cast<volatile tt_l1_ptr float*>(co + k * FP32_TILE);
                volatile tt_l1_ptr float* src = k < N_STREAMS ? src_p : src_c;
                const uint32_t col = k < N_STREAMS ? k : k - N_STREAMS;
                for (uint32_t row = 0; row < 32; ++row) {
                    dst[tile_off(row, 0)] = src[tile_off(row, col)];
                }
            }
            cb_push_back(CB_COEF, N_COEF);  // CB_RAW is reader-private scratch: never pushed
        }
#ifndef MPP_PRE
        cb_reserve_back(CB_Y, 1);
        noc_async_read(yg.get_noc_addr(r * CT + c), get_write_ptr(CB_Y), Y_PAGE);
#endif
        cb_reserve_back(CB_X, N_STREAMS);
        const uint32_t xw = get_write_ptr(CB_X);
        for (uint32_t i = 0; i < N_STREAMS; ++i) {  // stream i of the flat residual: column tile i CT + c
            noc_async_read(xg.get_noc_addr(r * N_STREAMS * CT + i * CT + c), xw + i * FP32_TILE, FP32_TILE);
        }
        noc_async_read_barrier();
#ifndef MPP_PRE
        cb_push_back(CB_Y, 1);
#endif
        cb_push_back(CB_X, N_STREAMS);
    }
}
