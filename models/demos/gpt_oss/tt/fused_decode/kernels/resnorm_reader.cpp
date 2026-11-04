// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash residual add + RMSNorm in one program (dflash/batched.py _ResNorm), reader (NCRISC). One core per
// (32-row tile m, NB-column-tile block nb) of the [rows, hidden] residual, block sharded on the GX x (rows / 32) grid
// (core (x = nb, y = m)). It replaces the all-reduce output's reshard to the residual, the residual add, the sharded
// norm and the norm output's copy to L1 interleaved (4 ops); with RN_ROUTER also the MoE router's matmul, with RN_X8
// the norm output's BFP8 copy (the grouped MoE's gather input). Pushes:
//   cb_r: this core's residual shard (tensor-backed: in place);
//   cb_y: the all-reduce output's NB tiles (m, nb NB + j) from its width shards ([rows, 32] per core: column tile c on
//         shard core c, tile m at offset m 2048);
//   cb_g: the norm weights' NB tiles (row 0 of faces 0 and 1 only: the row broadcast reads those);
//   cb_sc: the reduce scaler (1.0 in row 0 of each face);
//   cb_all: the row's GX partial row sums of squares (written into slot k by core k's writer, semaphore 0 counts them);
//   RN_ROUTER: cb_w: the router weights' NB K tiles (nb NB + j, 0) (BF16, [hidden, 32]); on the row's core 0 also
//   cb_lb: the router bias tile (every row the bias), cb_lall: the row's GX partial logits (semaphore 1 counts them).
//
// common runtime args: [y_addr, g_addr, (RN_ROUTER: w_addr, lb_addr), noc (x << 8 | y) of the all-reduce output's
// shard cores (2 per word)]
// RN_BIAS: cb_b: the bias's NB tiles (0, nb NB + j) of a [32, hidden] tensor (every row the bias).
// compile-time args: [cb_y, cb_r, cb_g, cb_sc, cb_all, NB, GX, cb_w, cb_lb, cb_lall, cb_b, accessors (g, (w, lb), (b))]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_all = get_compile_time_arg_val(4);
    constexpr uint32_t NB = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t cb_w = get_compile_time_arg_val(7);
    constexpr uint32_t cb_lb = get_compile_time_arg_val(8);
    constexpr uint32_t cb_lall = get_compile_time_arg_val(9);
    constexpr uint32_t cb_b = get_compile_time_arg_val(10);
    constexpr auto g_args = TensorAccessorArgs<11>();
#ifdef RN_ROUTER
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto lb_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t b_off = lb_args.next_compile_time_args_offset();
    constexpr uint32_t b_arg = 4;
#else
    constexpr uint32_t b_off = g_args.next_compile_time_args_offset();
    constexpr uint32_t b_arg = 2;
#endif
#ifdef RN_BIAS
    constexpr auto b_args = TensorAccessorArgs<b_off>();
    constexpr uint32_t noc_arg = b_arg + 1;
#else
    constexpr uint32_t noc_arg = b_arg;
#endif
    constexpr uint32_t tile = 2048;

    const uint32_t nb = get_absolute_logical_x(), m = get_absolute_logical_y();
    const auto s_g = TensorAccessor(g_args, g_addr, tile);

#ifndef RN_MPC
#define RN_MPC 1
#endif
    // RN_MPC row tiles per core (core y = m: row tiles m RN_MPC .. m RN_MPC + RN_MPC - 1, one after another)
    cb_reserve_back(cb_r, NB * RN_MPC);  // (the residual shard: in place)
    cb_push_back(cb_r, NB * RN_MPC);

    // the weights / scaler / bias / router tiles once (every row tile uses them)
    cb_reserve_back(cb_g, NB);
    const uint32_t gl1 = get_write_ptr(cb_g);
    for (uint32_t j = 0; j < NB; ++j) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(s_g.get_noc_addr(nb * NB + j) + f * 512, gl1 + j * tile + f * 512, 64);
        }
    }
    // the scaler: 1.0 (BF16) in row 0 of each face, zeros elsewhere
    cb_reserve_back(cb_sc, 1);
    const uint32_t sc = get_write_ptr(cb_sc);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, sc + z, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* s16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < 16; ++c) {
            s16[f * 256 + c] = 0x3F80;
        }
    }
    cb_push_back(cb_sc, 1);
    cb_push_back(cb_g, NB);
#ifdef RN_BIAS
    {
        const auto s_b = TensorAccessor(b_args, get_common_arg_val<uint32_t>(b_arg), tile);
        cb_reserve_back(cb_b, NB);
        const uint32_t bl1 = get_write_ptr(cb_b);
        for (uint32_t j = 0; j < NB; ++j) {
            noc_async_read(s_b.get_noc_addr(nb * NB + j), bl1 + j * tile, tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_b, NB);
    }
#endif
#ifdef RN_ROUTER
    {
        const auto s_w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2), tile);
        cb_reserve_back(cb_w, NB);
        const uint32_t wl1 = get_write_ptr(cb_w);
        for (uint32_t j = 0; j < NB; ++j) {
            noc_async_read(s_w.get_noc_addr(nb * NB + j), wl1 + j * tile, tile);
        }
        if (nb == 0) {
            const auto s_lb = TensorAccessor(lb_args, get_common_arg_val<uint32_t>(3), tile);
            cb_reserve_back(cb_lb, 1);
            noc_async_read(s_lb.get_noc_addr(0), get_write_ptr(cb_lb), tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, NB);
        if (nb == 0) {
            cb_push_back(cb_lb, 1);
        }
    }
#endif

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
#ifdef RN_ROUTER
    volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
#endif
    for (uint32_t mi = 0; mi < RN_MPC; ++mi) {
        const uint32_t rt = m * RN_MPC + mi;
        // the all-reduce output's NB tiles of row tile rt (RN_YPARTS: the output in parts of 8 row tiles, part p > 0's
        // address at common arg RN_YPART_ARG + p - 1)
#ifdef RN_YPARTS
        const uint32_t yp = rt / 8;
        const uint32_t y_row =
            (yp == 0 ? y_addr : get_common_arg_val<uint32_t>(RN_YPART_ARG + yp - 1)) + (rt % 8) * tile;
#else
        const uint32_t y_row = y_addr + rt * tile;
#endif
        cb_reserve_back(cb_y, NB);
        const uint32_t yl1 = get_write_ptr(cb_y);
        for (uint32_t j = 0; j < NB; ++j) {
            const uint32_t c = nb * NB + j;
            const uint32_t w = get_common_arg_val<uint32_t>(noc_arg + c / 2) >> (16 * (c % 2));
            noc_async_read(get_noc_addr((w >> 8) & 0xFF, w & 0xFF, y_row), yl1 + j * tile, tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_y, NB);
        // the row tile's partials (every core of the row writes slot mi GX + its x and increments semaphore 0: the
        // count reaches GX (mi + 1) once row tile mi's are all in)
        cb_reserve_back(cb_all, GX);
        while (*sem < GX * (mi + 1)) {
            invalidate_l1_cache();
        }
        cb_push_back(cb_all, GX);
#ifdef RN_ROUTER
        if (nb == 0) {  // the row tile's partial logits
            cb_reserve_back(cb_lall, GX);
            while (*sem1 < GX * (mi + 1)) {
                invalidate_l1_cache();
            }
            cb_push_back(cb_lall, GX);
        }
#endif
    }
}
