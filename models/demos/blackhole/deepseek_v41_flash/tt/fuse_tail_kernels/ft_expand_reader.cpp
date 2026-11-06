// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand reader (mhc_expand5_reader.cpp, token-group layout, y bf16 token rows) with y taken from the all-reduce
// landing buffers instead of DRAM: the x rows and the zero fill (independent of the all-reduce) are done first, then
// the reader waits for the landing semaphore (role 0: this core owns the landing buffer of its column slice, tokens of
// group 0 and 1 both read it: it signals the partner) or for the partner's signal (role 1: reads the chunk from the
// partner's landing buffer). y chunk format ('compact'): NG tiles x [face 0 rows 0..7 (256 B) | face 1 rows 0..7 (256
// B)], bf16, rows = tokens.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_b = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t NGRP = get_compile_time_arg_val(4);  // token groups of 4
    constexpr uint32_t NG = get_compile_time_arg_val(5);    // column tiles per core
    constexpr uint32_t SEM_P = get_compile_time_arg_val(6);
    constexpr uint32_t TG = get_compile_time_arg_val(
        7);  // tokens per group: 4 (T <= 8) or 6 (T <= 12): 4 TG x rows + TG y rows <= 32 rows of the B tile
    constexpr uint32_t YB =
        get_compile_time_arg_val(8);  // compact y tile bytes in the landing buffer: 512 (8 rows) or 1024 (16 rows)
    constexpr uint32_t PPL =
        get_compile_time_arg_val(9);  // reducers filling the landing buffer (semaphore increments per call)
    constexpr uint32_t P =
        get_compile_time_arg_val(10);  // passes: this core serves token groups g, g + NGRP, ... (T > 12)
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t G = P;              // B tiles per column tile: one per pass
    constexpr uint32_t YR0 = 4 * TG - 16;  // first y row inside face 2 / 3 of the B tile (rows 4 TG ..)

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t g0 = get_arg_val<uint32_t>(1);  // this core's first token group
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t partner_x = get_arg_val<uint32_t>(3);
    const uint32_t partner_y = get_arg_val<uint32_t>(4);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gsem = get_common_arg_val<uint32_t>(1);
    const uint32_t land_base = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    experimental::CB cbb(cb_b), cbs(cb_s);

    DeviceZoneScopedN("ER_ALL");
    cbs.reserve_back(1);
    cbb.reserve_back(G * NG);
    noc.async_write_zeros(cbb, G * NG * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    // pass p serves token group g0 + NGRP p (B tile gi * P + p); a group past the last token (nv = 0) leaves its zero
    // tile
    auto group_nv = [&](uint32_t p) -> uint32_t {
        const uint32_t tb_p = TG * (g0 + NGRP * p);
        return tb_p >= T ? 0 : ((T - tb_p) < TG ? (T - tb_p) : TG);
    };
    for (uint32_t gi = 0; gi < NG; ++gi) {
        const uint32_t j = j0 + gi;
        for (uint32_t p = 0; p < P; ++p) {
            const uint32_t tb_p = TG * (g0 + NGRP * p);
            const uint32_t nv_p = group_nv(p);
            const uint32_t bt = (gi * P + p) * TILE;
            for (uint32_t tl = 0; tl < nv_p; ++tl) {
                const uint32_t t = tb_p + tl;
                // token tl owns rows 4 tl .. 4 tl + 3 of the B tile: faces 0 / 1 for tl < 4, faces 2 / 3 above
                const uint32_t o0 = bt + (tl >> 2) * 2048 + (tl & 3) * 256;
                noc.async_read(x_acc, cbb, 256, {.page_id = t * NT + j, .offset_bytes = 0}, {.offset_bytes = o0});
                noc.async_read(
                    x_acc, cbb, 256, {.page_id = t * NT + j, .offset_bytes = 1024}, {.offset_bytes = o0 + 1024});
            }
        }
    }
    noc.async_read_barrier();

    // y from the all-reduce
    uint32_t y_l1;
    {
        DeviceZoneScopedN("ER_ywait");
        if (role == 0) {
            volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
            noc_semaphore_wait_min(gs, PPL);
            noc_semaphore_set(gs, 0);
            y_l1 = land_base;
            if constexpr (NGRP > 1) {
                noc_semaphore_inc(get_noc_addr(partner_x, partner_y, get_semaphore(SEM_P)), 1);
            }
        } else {
            volatile tt_l1_ptr uint32_t* ps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_P));
            noc_semaphore_wait_min(ps, 1);
            noc_semaphore_set(ps, 0);
            y_l1 = cbs.get_write_ptr();
            noc_async_read(get_noc_addr(partner_x, partner_y, land_base), y_l1, NG * YB);
            noc_async_read_barrier();
        }
    }
    {
        volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
        volatile tt_l1_ptr uint32_t* Y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(y_l1);
        for (uint32_t gi = 0; gi < NG; ++gi) {
            for (uint32_t p = 0; p < P; ++p) {
                const uint32_t tb_p = TG * (g0 + NGRP * p);
                const uint32_t nv_p = group_nv(p);
                volatile tt_l1_ptr uint32_t* Bt = B + (gi * P + p) * (TILE / 4);
                for (uint32_t h = 0; h < 2; ++h) {
                    volatile tt_l1_ptr uint32_t* src =
                        Y + gi * (YB / 4) + h * (YB / 8) + tb_p * 8;  // YB B per tile, YB / 2 per face, 32 B per row
                    volatile tt_l1_ptr uint32_t* dst =
                        Bt + 512 + 256 * h + YR0 * 16;  // y row m at row 4 TG + m (face 2 / 3, 64 B per row)
                    for (uint32_t m = 0; m < nv_p; ++m) {
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
    cbb.push_back(G * NG);
}
