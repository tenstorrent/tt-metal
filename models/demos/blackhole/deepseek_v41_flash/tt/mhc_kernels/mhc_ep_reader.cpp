// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand + next projection, reader.  Core (g, r): tokens [g*TG, (g+1)*TG), column tiles [r*TPJ, (r+1)*TPJ).
//   B1_j : row 4*tl + i = x[tok0 + tl, stream i, tile j]            (rows 0..3 of the token's stream tile, two 256 B
//   reads) B2_j : row tl = y row of token tok0 + tl, row TG + tl = y2 row  (token-row layout [1,1,T,D] only; bf16 y is
//   widened here) A1   : A1[4*tl + j, 4*tl + i] = comb[t][i][j]                   A2: A2[4*tl + j, tl] = post[t][j] (+
//   A2[4*tl + j, TG + tl] if y2)
// so that  X'_j = A1 @ B1_j + A2 @ B2_j  has row 4*tl + j = new stream j of token tl (the projection's input layout).
// Also loads the 4*TPJ projection weight tiles of the core's column range.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define WI(r, c) ((((r) >> 4) << 9) + (((c) >> 4) << 8) + (((r) & 15) << 4) + ((c) & 15))

void kernel_main() {
    constexpr uint32_t cb_b1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s = get_compile_time_arg_val(4);  // scratch: comb/post rows, raw bf16 y rows
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t TG = get_compile_time_arg_val(6);
    constexpr uint32_t NT = get_compile_time_arg_val(7);
    constexpr uint32_t TPJ = get_compile_time_arg_val(8);
    constexpr uint32_t Y_BF16 = get_compile_time_arg_val(9);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<y2_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t YB = Y_BF16 ? 2048 : TILE;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t tok0 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t y2_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(5);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, YB);
    const auto y2_acc = TensorAccessor(y2_args, y2_addr, TILE);
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbb1(cb_b1), cbb2(cb_b2), cba(cb_a), cbw(cb_w), cbs(cb_s);

    // scratch layout (bytes): [0, TG*512): comb/post rows of token tl at tl*512 (comb rows 0..3: 256 B, post rows 0..3:
    // 256 B);
    //                         [TG*512, ...): raw bf16 y rows, slot (jj*2 + h) of TG*32 B
    constexpr uint32_t YS = TG * 512;
    cbs.reserve_back(1);
    cbb1.reserve_back(TPJ);
    cbb2.reserve_back(TPJ);
    cba.reserve_back(2);
    cbw.reserve_back(4 * TPJ);

    for (uint32_t tl = 0; tl < TG; ++tl) {
        noc.async_read(c_acc, cbs, 256, {.page_id = tok0 + tl, .offset_bytes = 0}, {.offset_bytes = tl * 512});
        noc.async_read(p_acc, cbs, 256, {.page_id = tok0 + tl, .offset_bytes = 0}, {.offset_bytes = tl * 512 + 256});
    }
    if constexpr (Y_BF16) {
        // bf16 tile page: face h (cols 16h..), row m at h*512 + m*32; rows tok0..tok0+TG-1 are contiguous inside a face
        const uint32_t base = ((tok0 >> 4) << 10) + ((tok0 & 15) << 5);
        for (uint32_t jj = 0; jj < TPJ; ++jj) {
            noc.async_read(
                y_acc,
                cbs,
                TG * 32,
                {.page_id = r * TPJ + jj, .offset_bytes = base},
                {.offset_bytes = YS + (jj * 2) * TG * 32});
            noc.async_read(
                y_acc,
                cbs,
                TG * 32,
                {.page_id = r * TPJ + jj, .offset_bytes = base + 512},
                {.offset_bytes = YS + (jj * 2 + 1) * TG * 32});
        }
    }
    noc.async_write_zeros(cba, 2 * TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbb1, TPJ * TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbb2, TPJ * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t jj = 0; jj < TPJ; ++jj) {
        const uint32_t j = r * TPJ + jj;
        for (uint32_t tl = 0; tl < TG; ++tl) {
            const uint32_t page = (tok0 + tl) * NT + j;
            noc.async_read(
                x_acc, cbb1, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = jj * TILE + RO(4 * tl)});
            noc.async_read(
                x_acc,
                cbb1,
                256,
                {.page_id = page, .offset_bytes = 1024},
                {.offset_bytes = jj * TILE + 1024 + RO(4 * tl)});
        }
        if constexpr (!Y_BF16) {
            noc.async_read(y_acc, cbb2, TG * 64, {.page_id = j, .offset_bytes = RO(tok0)}, {.offset_bytes = jj * TILE});
            noc.async_read(
                y_acc,
                cbb2,
                TG * 64,
                {.page_id = j, .offset_bytes = 1024 + RO(tok0)},
                {.offset_bytes = jj * TILE + 1024});
        }
        if constexpr (HAS_Y2) {
            noc.async_read(
                y2_acc, cbb2, TG * 64, {.page_id = j, .offset_bytes = RO(tok0)}, {.offset_bytes = jj * TILE + RO(TG)});
            noc.async_read(
                y2_acc,
                cbb2,
                TG * 64,
                {.page_id = j, .offset_bytes = 1024 + RO(tok0)},
                {.offset_bytes = jj * TILE + 1024 + RO(TG)});
        }
    }
    for (uint32_t n = 0; n < 4 * TPJ; ++n) {
        noc.async_read(w_acc, cbw, TILE, {.page_id = r * 4 * TPJ + n, .offset_bytes = 0}, {.offset_bytes = n * TILE});
    }
    noc.async_read_barrier();

    // ---- A tiles ----
    {
        volatile tt_l1_ptr uint32_t* A1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
        volatile tt_l1_ptr uint32_t* A2 = A1 + TILE / 4;
        volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t tl = 0; tl < TG; ++tl) {
            volatile tt_l1_ptr uint32_t* sc = S + tl * 128;
            volatile tt_l1_ptr uint32_t* sp = sc + 64;
            for (uint32_t j = 0; j < 4; ++j) {
                for (uint32_t i = 0; i < 4; ++i) {
                    A1[WI(4 * tl + j, 4 * tl + i)] = sc[i * 16 + j];
                }
                A2[WI(4 * tl + j, tl)] = sp[j * 16];
                if constexpr (HAS_Y2) {
                    A2[WI(4 * tl + j, TG + tl)] = sp[j * 16];
                }
            }
        }
        if constexpr (Y_BF16) {
            volatile tt_l1_ptr uint32_t* B2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb2.get_write_ptr());
            for (uint32_t jj = 0; jj < TPJ; ++jj) {
                for (uint32_t h = 0; h < 2; ++h) {
                    volatile tt_l1_ptr uint32_t* src = S + (YS >> 2) + (jj * 2 + h) * TG * 8;
                    volatile tt_l1_ptr uint32_t* dst = B2 + jj * (TILE / 4) + 256 * h;  // rows 0..TG-1 of face (0 | 1)
                    for (uint32_t m = 0; m < TG; ++m) {
                        // TG <= 8: rows m < 16 -> face 0 / face 1 columns of rows 0..; face 0 holds rows 0..15
                        for (uint32_t e = 0; e < 8; ++e) {
                            const uint32_t w = src[m * 8 + e];
                            dst[m * 16 + 2 * e] = w << 16;
                            dst[m * 16 + 2 * e + 1] = w & 0xFFFF0000u;
                        }
                    }
                }
            }
        }
    }
    cba.push_back(2);
    cbb1.push_back(TPJ);
    cbb2.push_back(TPJ);
    cbw.push_back(4 * TPJ);
}
