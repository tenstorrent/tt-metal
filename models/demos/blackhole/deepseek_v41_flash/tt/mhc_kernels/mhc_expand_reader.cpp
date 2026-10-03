// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand, reader.  new_j = post_j * y + sum_i comb[i,j] * x_i  as ONE tile matmul per (token, 32-col tile):
//   out(4 x 32) = A_t(4 x 5) @ B(5 x 32),  A_t[j][i] = comb[t,i,j], A_t[j][4] = post[t,j],  B = [x rows 0..3 ; y row].
// The reader builds the 4 A_t tiles once (fp32, face 0) and gathers B tiles (rows 0..3 of the x tile, row 4 = y).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);  // scratch: 2 pages
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);     // 32-col tiles per row (D/32)
    constexpr uint32_t Y_ROW = get_compile_time_arg_val(5);  // y is [1,1,T,D] (token rows) instead of [T,1,1,D]
    constexpr uint32_t Y_BF16 = get_compile_time_arg_val(6);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(7);  // second term y2 [1,1,T,D] fp32
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<y2_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t y2_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, Y_BF16 ? 2048 : TILE);
    const auto y2_acc = TensorAccessor(y2_args, y2_addr, TILE);
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cba(cb_a), cbb(cb_b), cbs(cb_s);

    // ---- A tiles ----
    cbs.reserve_back(2);
    cba.reserve_back(T);
    noc.async_write_zeros(cba, T * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
    volatile tt_l1_ptr uint32_t* sp = sc + TILE / 4;
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(c_acc, cbs, 256, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(p_acc, cbs, 256, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = TILE});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* At = A + t * (TILE / 4);
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t i = 0; i < 4; ++i) {
                At[j * 16 + i] = sc[i * 16 + j];  // A[j][i] = comb[i][j]
            }
            At[j * 16 + 4] = sp[j * 16];  // A[j][4] = post[j]
            if (HAS_Y2) {
                At[j * 16 + 5] = sp[j * 16];  // A[j][5] = post[j]
            }
        }
    }
    cba.push_back(T);

    // ---- B tiles: (j, t) order ----
    const uint32_t ng = j1 - j0;
    const uint32_t nb = ng * T;
    cbb.reserve_back(nb);
    noc.async_write_zeros(cbb, nb * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t gi = 0; gi < ng; ++gi) {
        const uint32_t j = j0 + gi;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t dst = (gi * T + t) * TILE;
            const uint32_t page = t * NT + j;
            noc.async_read(x_acc, cbb, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = dst});
            noc.async_read(x_acc, cbb, 256, {.page_id = page, .offset_bytes = 1024}, {.offset_bytes = dst + 1024});
            // y row -> B row 4
            const uint32_t ypage = Y_ROW ? j : page;
            const uint32_t yrow = Y_ROW ? t : 0;
            if (Y_BF16) {
                // 64-byte aligned reads of the row pair holding `yrow` (bf16 rows are 32 B), expanded to fp32 below
                const uint32_t sbase = (gi * T + t) * 128;
                noc.async_read(
                    y_acc,
                    cbs,
                    64,
                    {.page_id = ypage, .offset_bytes = ((yrow >> 4) << 10) + (((yrow & 15) >> 1) << 6)},
                    {.offset_bytes = sbase});
                noc.async_read(
                    y_acc,
                    cbs,
                    64,
                    {.page_id = ypage, .offset_bytes = 512 + ((yrow >> 4) << 10) + (((yrow & 15) >> 1) << 6)},
                    {.offset_bytes = sbase + 64});
            } else {
                noc.async_read(
                    y_acc, cbb, 64, {.page_id = ypage, .offset_bytes = RO(yrow)}, {.offset_bytes = dst + 4 * 64});
                noc.async_read(
                    y_acc,
                    cbb,
                    64,
                    {.page_id = ypage, .offset_bytes = 1024 + RO(yrow)},
                    {.offset_bytes = dst + 1024 + 4 * 64});
            }
            if (HAS_Y2) {
                noc.async_read(y2_acc, cbb, 64, {.page_id = j, .offset_bytes = RO(t)}, {.offset_bytes = dst + 5 * 64});
                noc.async_read(
                    y2_acc,
                    cbb,
                    64,
                    {.page_id = j, .offset_bytes = 1024 + RO(t)},
                    {.offset_bytes = dst + 1024 + 5 * 64});
            }
        }
    }
    noc.async_read_barrier();
    if (Y_BF16) {
        volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
        volatile tt_l1_ptr uint16_t* S16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbs.get_write_ptr());
        for (uint32_t gi = 0; gi < ng; ++gi) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t yrow = Y_ROW ? t : 0;
                const uint32_t sbase = (gi * T + t) * 64;  // in uint16 units: 128 B per slot
                volatile tt_l1_ptr uint32_t* Bt = B + (gi * T + t) * (TILE / 4);
                for (uint32_t e = 0; e < 16; ++e) {
                    Bt[4 * 16 + e] = ((uint32_t)S16[sbase + (yrow & 1) * 16 + e]) << 16;
                    Bt[256 + 4 * 16 + e] = ((uint32_t)S16[sbase + 32 + (yrow & 1) * 16 + e]) << 16;
                }
            }
        }
    }
    cbb.push_back(nb);
}
