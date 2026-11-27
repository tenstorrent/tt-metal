// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix — reader (BRISC, NoC0): y (+ y2) tiles, the residual tiles (flat page order s * 128 + c, or the
// stacked [T,1,4,4096] residual re-laid out row by row), the T coefficient rows -> 20 column-0 tiles.
// Compile-time args: X_MODE (0 flat, 1 stacked), ADD, TensorAccessorArgs of y, y2, x, coef.
// Runtime args: y, y2, x, coef addresses, first column tile c0, tokens T.
// Megakernel (MK_POST_WAIT_SEM / MK_POST_WAIT_COUNT): the residual and the coefficients are read first, then the
// reader waits for the sublayer output's producers before reading y (+ y2).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace mpost;

void kernel_main() {
    constexpr uint32_t X_MODE = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t ADD = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + 2>();
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<y2_args.next_compile_time_args_offset()>();
    constexpr auto coef_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto y2 = TensorAccessor(y2_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto coef = TensorAccessor(coef_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const uint32_t c0 = get_arg_val<uint32_t>(ph_rt0 + 4);
    const uint32_t T = get_arg_val<uint32_t>(ph_rt0 + 5);

    DeviceZoneScopedN("MP-RD");  // device-profiler zone (no-op without the profiler)
    cb_reserve_back(CB_X, X_TILES);
    const uint32_t x_l1 = get_write_ptr(CB_X);
    if constexpr (X_MODE == 0) {
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                noc_async_read_page(s * COL_TILES + c0 + j, x, x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE);
            }
        }
    } else {
        // stacked [T,1,4,4096]: flat tile (s, j) row t = stacked page t * 128 + c0 + j row s (64-B face rows)
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < X_TILES * FP32_TILE; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                const uint32_t page = t * COL_TILES + c0 + j;
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    const uint32_t dst = x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE;
                    noc_async_read(x.get_noc_addr(page, tile_off(s, 0) * 4), dst + tile_off(t, 0) * 4, 64);
                    noc_async_read(x.get_noc_addr(page, tile_off(s, 16) * 4), dst + tile_off(t, 16) * 4, 64);
                }
            }
        }
    }
#if defined(MPOST_V2)
    // Stage 14c v2: ONE compact coefficient tile — coefficient k of token t at plane k, lane 8 (t % 4) + t / 4
    // (mhc_post_v2_sfpu.hpp); zeroed first (tokens >= T, planes >= 20)
    cb_reserve_back(CB_COEF, 1);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < FP32_TILE; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, get_write_ptr(CB_COEF) + off, MEM_ZEROS_SIZE);
        }
    }
    cb_reserve_back(CB_CROW, 1);
    {
        const uint32_t crow = get_write_ptr(CB_CROW);
#if defined(MPOST_COEF_PLANE)
        // "cp": mhc_pre's Sinkhorn page as is — coefficient k of token t at plane N_STREAMS + k, lane t
        noc_async_read(coef.get_noc_addr(0), crow, FP32_TILE);
#else
        for (uint32_t t = 0; t < T; ++t) {
            noc_async_read(coef.get_noc_addr(0, tile_off(t, 0) * 4), crow + tile_off(t, 0) * 4, 64);
            noc_async_read(coef.get_noc_addr(0, tile_off(t, 16) * 4), crow + tile_off(t, 16) * 4, 64);
        }
#endif
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(crow);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_COEF));
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < N_COEF; ++k) {
#if defined(MPOST_COEF_PLANE)
                dst[plane_off(k, 8 * (t & 3) + (t >> 2))] = src[plane_off(N_STREAMS + k, t)];
#else
                dst[plane_off(k, 8 * (t & 3) + (t >> 2))] = src[tile_off(t, k)];
#endif
            }
        }
    }
    cb_push_back(CB_X, X_TILES);
    cb_push_back(CB_COEF, 1);
#else
    // the 20 column-0 coefficient tiles: zeroed whole first (NoC reads of the core's zero region) — the SFPU column
    // broadcast also loads columns 1..7 of faces 0 / 2 and masks them with a 0.0 multiply, so stale Inf / NaN bits
    // there would turn into NaN rows (review M1)
    cb_reserve_back(CB_COEF, N_COEF);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t base = get_write_ptr(CB_COEF);
        for (uint32_t off = 0; off < N_COEF * FP32_TILE; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
    }
    // coefficient rows t < T (cols 0..19 = 16 in face 0 + 4 in face 1)
    cb_reserve_back(CB_CROW, 1);
    const uint32_t crow = get_write_ptr(CB_CROW);
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read(coef.get_noc_addr(0, tile_off(t, 0) * 4), crow + tile_off(t, 0) * 4, 64);
        noc_async_read(coef.get_noc_addr(0, tile_off(t, 16) * 4), crow + tile_off(t, 16) * 4, 64);
    }
    noc_async_read_barrier();  // zeroing + coefficient rows landed before the column-0 stores below
    // 20 column-0 tiles: rows t < T = coefficient k of token t, rows >= T zero
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(crow);
        for (uint32_t k = 0; k < N_COEF; ++k) {
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_COEF) + k * FP32_TILE);
            for (uint32_t r = 0; r < 32; ++r) {
                dst[tile_off(r, 0)] = r < T ? src[tile_off(r, k)] : 0u;
            }
        }
    }
    cb_push_back(CB_X, X_TILES);
    cb_push_back(CB_COEF, N_COEF);
#endif  // MPOST_V2
#if defined(MK_POST_WAIT_SEM)
#if defined(MK_POST_SECOND) && defined(MK_POST2_WAIT_COUNT)
    mk::wait_handoff(MK_POST_WAIT_SEM, MK_POST2_WAIT_COUNT);  // the FFN instance's own producer count (owner kernel)
#else
    mk::wait_handoff(MK_POST_WAIT_SEM, MK_POST_WAIT_COUNT);
#endif
#endif
    cb_reserve_back(CB_Y, COLS_PER_CORE);
    const uint32_t y_page = get_tile_size(CB_Y);
#if defined(MK_POST_Y_FABRIC)
    {
        // TP (t > 4): y must be the all-reduced sum, not this chip's local partial —
        // y = the in-program fabric all-reduce's result in the lanes' result buffer (tt/fabric_ar.py): column tile c of
        // lane c / 64 at res + (c % 64) * V; B <= 16: face-0 rows t < B then face-1 rows (V = 2 x CH bytes), B > 16:
        // the whole bf16 tile (V = the page). Runtime args at MK_POST_FAB_RT (absolute; else ph_rt0 + 6): lane 0
        // (x, y), lane 1 (x, y), res, V, CH (FabricPhase.consumer_args, as reader_p.cpp).
#if defined(MK_POST_FAB_RT)
        const uint32_t fab = MK_POST_FAB_RT;
#else
        const uint32_t fab = ph_rt0 + 6;
#endif
        const uint32_t res = get_arg_val<uint32_t>(fab + 4);
        const uint32_t fv = get_arg_val<uint32_t>(fab + 5), fch = get_arg_val<uint32_t>(fab + 6);
        const uint32_t base = get_write_ptr(CB_Y);
        if (fv < y_page) {  // packed rows: the padding rows (t >= B) and faces 2 / 3 of the pages zero
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t off = 0; off < COLS_PER_CORE * y_page; off += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
        }
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            const uint32_t c = c0 + j, lane = c / 64;
            const uint64_t src = get_noc_addr(
                get_arg_val<uint32_t>(fab + 2 * lane), get_arg_val<uint32_t>(fab + 1 + 2 * lane), res + (c % 64) * fv);
            if (fv >= y_page) {
                noc_async_read(src, base + j * y_page, y_page);
            } else {
                noc_async_read(src, base + j * y_page, fch);
                noc_async_read(src + fch, base + j * y_page + 512, fch);  // bf16 face 1
            }
        }
    }
#else
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        noc_async_read_page(c0 + j, y, get_write_ptr(CB_Y) + j * y_page);
    }
#endif
    if constexpr (ADD) {
        cb_reserve_back(CB_Y2, COLS_PER_CORE);
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_read_page(c0 + j, y2, get_write_ptr(CB_Y2) + j * get_tile_size(CB_Y2));
        }
    }
    noc_async_read_barrier();
    cb_push_back(CB_Y, COLS_PER_CORE);
    if constexpr (ADD) {
        cb_push_back(CB_Y2, COLS_PER_CORE);
    }
}
