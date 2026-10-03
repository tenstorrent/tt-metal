// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand, reader.  new_j = post_j * (y [+ y2]) + sum_i comb[i,j] * x_i.
// Tokens are processed in groups of 4: per (group g, column tile j) ONE B tile
//   rows 4*tl + i (tl = token in group, i = stream)   : x rows 0..3 of token page (t, j)      (2 reads of 256 B)
//   rows 16 + tl                                       : y  row of token t                    (1 read per face half per
//   group) rows 20 + tl                                       : y2 row of token t
// is built; the compute kernel multiplies it by a per-token A tile (see the writer kernel, which builds the A tiles).
// All DRAM reads are issued up front and waited for with a single barrier.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_b = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t Y_ROW = get_compile_time_arg_val(4);
    constexpr uint32_t Y_BF16 = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);   // token groups of 4
    constexpr uint32_t NG = get_compile_time_arg_val(8);  // column tiles per core
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t y2_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, Y_BF16 ? 2048 : TILE);
    const auto y2_acc = TensorAccessor(y2_args, y2_addr, TILE);
    experimental::CB cbb(cb_b), cbs(cb_s);

    cbs.reserve_back(1);
    cbb.reserve_back(G * NG);

    // bf16 y: stage the raw bf16 rows in scratch (issued before the zero fill so the DRAM latency overlaps it)
    if constexpr (Y_BF16) {
        if constexpr (Y_ROW) {
            for (uint32_t gi = 0; gi < NG; ++gi) {
                for (uint32_t g = 0; g < G; ++g) {
                    const uint32_t r0 = 4 * g;
                    const uint32_t base = ((r0 >> 4) << 10) + ((r0 & 15) << 5);
                    const uint32_t slot = (g * NG + gi) * 2;
                    noc.async_read(
                        y_acc, cbs, 128, {.page_id = j0 + gi, .offset_bytes = base}, {.offset_bytes = slot * 128});
                    noc.async_read(
                        y_acc,
                        cbs,
                        128,
                        {.page_id = j0 + gi, .offset_bytes = base + 512},
                        {.offset_bytes = (slot + 1) * 128});
                }
            }
        } else {
            for (uint32_t gi = 0; gi < NG; ++gi) {
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t slot = (gi * T + t) * 2;
                    noc.async_read(
                        y_acc, cbs, 64, {.page_id = t * NT + j0 + gi, .offset_bytes = 0}, {.offset_bytes = slot * 128});
                    noc.async_read(
                        y_acc,
                        cbs,
                        64,
                        {.page_id = t * NT + j0 + gi, .offset_bytes = 512},
                        {.offset_bytes = (slot + 1) * 128});
                }
            }
        }
    }

    noc.async_write_zeros(cbb, G * NG * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    // x rows (and fp32 y / y2 rows) straight into the B tiles
    for (uint32_t gi = 0; gi < NG; ++gi) {
        const uint32_t j = j0 + gi;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t bt = (gi * G + g) * TILE;
            const uint32_t nv = (T - 4 * g) < 4 ? (T - 4 * g) : 4;
            for (uint32_t tl = 0; tl < nv; ++tl) {
                const uint32_t t = 4 * g + tl;
                noc.async_read(
                    x_acc, cbb, 256, {.page_id = t * NT + j, .offset_bytes = 0}, {.offset_bytes = bt + tl * 256});
                noc.async_read(
                    x_acc,
                    cbb,
                    256,
                    {.page_id = t * NT + j, .offset_bytes = 1024},
                    {.offset_bytes = bt + 1024 + tl * 256});
            }
            if constexpr (!Y_BF16) {
                if constexpr (Y_ROW) {
                    noc.async_read(
                        y_acc, cbb, nv * 64, {.page_id = j, .offset_bytes = RO(4 * g)}, {.offset_bytes = bt + 2048});
                    noc.async_read(
                        y_acc,
                        cbb,
                        nv * 64,
                        {.page_id = j, .offset_bytes = 1024 + RO(4 * g)},
                        {.offset_bytes = bt + 3072});
                } else {
                    for (uint32_t tl = 0; tl < nv; ++tl) {
                        const uint32_t t = 4 * g + tl;
                        noc.async_read(
                            y_acc,
                            cbb,
                            64,
                            {.page_id = t * NT + j, .offset_bytes = 0},
                            {.offset_bytes = bt + 2048 + tl * 64});
                        noc.async_read(
                            y_acc,
                            cbb,
                            64,
                            {.page_id = t * NT + j, .offset_bytes = 1024},
                            {.offset_bytes = bt + 3072 + tl * 64});
                    }
                }
            }
            if constexpr (HAS_Y2) {
                noc.async_read(
                    y2_acc, cbb, nv * 64, {.page_id = j, .offset_bytes = RO(4 * g)}, {.offset_bytes = bt + 2048 + 256});
                noc.async_read(
                    y2_acc,
                    cbb,
                    nv * 64,
                    {.page_id = j, .offset_bytes = 1024 + RO(4 * g)},
                    {.offset_bytes = bt + 3072 + 256});
            }
        }
    }
    noc.async_read_barrier();

    if constexpr (Y_BF16) {
        volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
        volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t gi = 0; gi < NG; ++gi) {
            for (uint32_t g = 0; g < G; ++g) {
                const uint32_t nv = (T - 4 * g) < 4 ? (T - 4 * g) : 4;
                volatile tt_l1_ptr uint32_t* Bt = B + (gi * G + g) * (TILE / 4);
                for (uint32_t h = 0; h < 2; ++h) {
                    if constexpr (Y_ROW) {
                        volatile tt_l1_ptr uint32_t* src = S + ((g * NG + gi) * 2 + h) * 32;  // 128 B slot = 32 words
                        volatile tt_l1_ptr uint32_t* dst = Bt + 512 + 256 * h;
                        for (uint32_t m = 0; m < nv; ++m) {
                            for (uint32_t e = 0; e < 8; ++e) {
                                const uint32_t w = src[m * 8 + e];
                                dst[m * 16 + 2 * e] = w << 16;
                                dst[m * 16 + 2 * e + 1] = w & 0xFFFF0000u;
                            }
                        }
                    } else {
                        for (uint32_t tl = 0; tl < nv; ++tl) {
                            const uint32_t t = 4 * g + tl;
                            volatile tt_l1_ptr uint32_t* src = S + ((gi * T + t) * 2 + h) * 32;
                            volatile tt_l1_ptr uint32_t* dst = Bt + 512 + 256 * h + tl * 16;
                            for (uint32_t e = 0; e < 8; ++e) {
                                const uint32_t w = src[e];
                                dst[2 * e] = w << 16;
                                dst[2 * e + 1] = w & 0xFFFF0000u;
                            }
                        }
                    }
                }
            }
        }
    }
    cbb.push_back(G * NG);
}
