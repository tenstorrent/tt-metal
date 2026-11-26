// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix, PACKED decode mode (T <= 4) — reader (BRISC, NoC0). The core's whole post-mix fits in ONE tile:
// row r = (s * 2 + j) * T + t (output stream s, column tile j = c0 + 0/1, token t; 8 T <= 32 rows), so the compute does
// 5 broadcast products + 4 sums on one tile instead of 2 x 4 x (5 + 4) full-tile ops (mhc_post_common.hpp: same fp32
// SFPU arithmetic in the same order, bitwise the unpacked program). This reader builds the packed operands:
//   X_i  (CB_X pages 0..3)  row r = residual stream i, token t, column tile j   (replicated over s)
//   Y    (CB_Y page 0)      row r = y[t, column tile j]                          (+ Y2, replicated over s)
//   COEF (CB_COEF pages 0..4, column 0): page 0 row r = post_s(t), page 1 + i row r = C[i][s](t)
// from whole source pages read into the scratch pages behind them (CB_X pages 4..11, CB_Y / CB_Y2 pages 1..2, CB_CROW)
// and RISC row copies; every other row stays zero (the CBs are zeroed first).
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

namespace {

constexpr uint32_t X_RAW = N_STREAMS;  // CB_X page of the first raw source page

// one 32-value row of a tile (two 16-value face rows) between L1 tiles; esize = element bytes (4 fp32, 2 bf16)
FORCE_INLINE void copy_row(uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row, uint32_t esize) {
    const uint32_t words = 16 * esize / 4;
    for (uint32_t h = 0; h < 2; ++h) {
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + tile_off(src_row, 16 * h) * esize);
        volatile tt_l1_ptr uint32_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + tile_off(dst_row, 16 * h) * esize);
        for (uint32_t w = 0; w < words; ++w) {
            d[w] = s[w];
        }
    }
}

FORCE_INLINE void zero(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
    }
}

}  // namespace

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
    DeviceZoneScopedN("MP-RD");

    // the packed operands start as zero tiles; the raw residual pages and the coefficient rows land behind them
    cb_reserve_back(CB_X, X_RAW + 2 * N_STREAMS);
    cb_reserve_back(CB_COEF, 1 + N_STREAMS);
    cb_reserve_back(CB_CROW, 1);
    const uint32_t xl1 = get_write_ptr(CB_X), cl1 = get_write_ptr(CB_COEF), crow = get_write_ptr(CB_CROW);
    zero(xl1, X_RAW * FP32_TILE);
    zero(cl1, (1 + N_STREAMS) * FP32_TILE);
    // raw residual pages: flat page s * 128 + c0 + j -> raw (s, j); stacked page t * 128 + c0 + j -> raw (t, j)
    for (uint32_t a = 0; a < (X_MODE == 0 ? N_STREAMS : T); ++a) {
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_read_page(a * COL_TILES + c0 + j, x, xl1 + (X_RAW + a * COLS_PER_CORE + j) * FP32_TILE);
        }
    }
    for (uint32_t t = 0; t < T; ++t) {  // coefficient rows t < T (cols 0..19 = 16 in face 0 + 4 in face 1)
        noc_async_read(coef.get_noc_addr(0, tile_off(t, 0) * 4), crow + tile_off(t, 0) * 4, 64);
        noc_async_read(coef.get_noc_addr(0, tile_off(t, 16) * 4), crow + tile_off(t, 16) * 4, 64);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* cr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(crow);
    for (uint32_t s = 0; s < N_STREAMS; ++s) {
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t r = (s * COLS_PER_CORE + j) * T + t;
                for (uint32_t i = 0; i < N_STREAMS; ++i) {
                    const uint32_t src = X_MODE == 0 ? xl1 + (X_RAW + i * COLS_PER_CORE + j) * FP32_TILE
                                                     : xl1 + (X_RAW + t * COLS_PER_CORE + j) * FP32_TILE;
                    copy_row(src, X_MODE == 0 ? t : i, xl1 + i * FP32_TILE, r, 4);
                }
                volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cl1);
                c[tile_off(r, 0)] = cr[tile_off(t, s)];  // post_s
                for (uint32_t i = 0; i < N_STREAMS; ++i) {
                    c[(1 + i) * (FP32_TILE / 4) + tile_off(r, 0)] = cr[tile_off(t, N_STREAMS + N_STREAMS * i + s)];
                }
            }
        }
    }
    cb_push_back(CB_X, N_STREAMS);
    cb_push_back(CB_COEF, 1 + N_STREAMS);

#if defined(MK_POST_WAIT_SEM)
#if defined(MK_POST_SECOND) && defined(MK_POST2_WAIT_COUNT)
    mk::wait_handoff(MK_POST_WAIT_SEM, MK_POST2_WAIT_COUNT);  // Stage 14 TP sparse layer: the FFN instance's producers
#else
    mk::wait_handoff(MK_POST_WAIT_SEM, MK_POST_WAIT_COUNT);
#endif
#endif
    // y (+ y2): raw pages c0 + j behind the packed tile, then the rows
    constexpr uint32_t NY = ADD ? 2 : 1;
    for (uint32_t n = 0; n < NY; ++n) {
        const uint32_t cb = n == 0 ? CB_Y : CB_Y2;
        const uint32_t page = get_tile_size(cb);
        cb_reserve_back(cb, 1 + COLS_PER_CORE);
        const uint32_t l1 = get_write_ptr(cb);
        zero(l1, page);
#if defined(MK_POST_Y_FABRIC)
        // Stage 14 TP8: y = the in-program fabric all-reduce's result (tt/fabric_ar.py), packed in the lanes' result
        // buffer: column tile c of lane c / 64 at res + (c % 64) * V, face-0 rows t < T then face-1 rows (only those
        // rows of the raw page are read below). Runtime args at MK_POST_FAB_RT (absolute; default ph_rt0 + 6, i.e.
        // appended right after this phase's own): lane 0 (x, y), lane 1 (x, y), res, V, CH.
#if defined(MK_POST_FAB_RT)
        const uint32_t fab = MK_POST_FAB_RT;
#else
        const uint32_t fab = ph_rt0 + 6;
#endif
        if (n == 0) {
            const uint32_t res = get_arg_val<uint32_t>(fab + 4);
            const uint32_t fv = get_arg_val<uint32_t>(fab + 5), fch = get_arg_val<uint32_t>(fab + 6);
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                const uint32_t c = c0 + j, lane = c / 64;
                const uint32_t lx = get_arg_val<uint32_t>(fab + 2 * lane);
                const uint32_t ly = get_arg_val<uint32_t>(fab + 1 + 2 * lane);
                const uint64_t src = get_noc_addr(lx, ly, res + (c % 64) * fv);
                noc_async_read(src, l1 + (1 + j) * page, fch);
                noc_async_read(src + fch, l1 + (1 + j) * page + 512, fch);
            }
            continue;
        }
#endif
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_read_page(c0 + j, n == 0 ? y : y2, l1 + (1 + j) * page);
        }
    }
    noc_async_read_barrier();
    for (uint32_t n = 0; n < NY; ++n) {
        const uint32_t cb = n == 0 ? CB_Y : CB_Y2;
        const uint32_t page = get_tile_size(cb), esize = page / 1024;
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                for (uint32_t t = 0; t < T; ++t) {
                    copy_row(l1 + (1 + j) * page, t, l1, (s * COLS_PER_CORE + j) * T + t, esize);
                }
            }
        }
        cb_push_back(cb, 1);
    }
}
