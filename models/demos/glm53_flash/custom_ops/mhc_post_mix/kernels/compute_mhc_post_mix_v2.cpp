// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC residual update, exact fp32 (v2: resident operands, lane-layout coefficient tiles, hand-written
// SFPU mix). Per token-tile r the n*n + n coefficients arrive either as the raw Sinkhorn tiles (EXPAND: comb
// [32 x n*n], post [32 x n] in CB_RAW; this kernel expands them once into the n + 1 lane-layout tiles of
// CB_COEF -- stream j's tile holds comb[:, i*n+j] in column group i, the post tile holds post[:, j] in column
// group j; group m = face m/2, column parity m%2, see mhc_sfpu_x8.h) or already expanded (CB_COEF filled by
// the reader). Per position (r, c) the n stream tiles, y (fp32 or bf16,
// widened on unpack) and the post tile are copied once into the fp32 dest (full-sync mode: 8 resident slots),
// then for every output stream j the comb tile of j is copied in and
//     out = sum_i comb[i,j] * x_i + post[j] * y
// is evaluated by mhc_sfpu::stream_mix (per-row coefficient slots straight from the lane-layout tiles, FMAs),
// packed, and the dest released WITHOUT clearing so the operands stay for the next stream. With ``collapse``
// the n streams accumulate into one tile scaled by 1/n.
// Dest slots: 0..n-1 = x_i, n = y, n+1 = comb tile of the current stream, n+2 = post, n+3 = out (n = 4 -> 8).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "mhc_sfpu_x8.h"

namespace {

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_Y = get_compile_time_arg_val(1);
constexpr uint32_t CB_RAW = get_compile_time_arg_val(2);
constexpr uint32_t CB_COEF = get_compile_time_arg_val(3);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(4);
constexpr uint32_t N = get_compile_time_arg_val(5);
constexpr uint32_t CT = get_compile_time_arg_val(6);
constexpr bool COLLAPSE = get_compile_time_arg_val(7) != 0;
constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(8);
constexpr bool EXPAND = get_compile_time_arg_val(9) != 0;
// HMODE: the pre-mix h = sum_i pre_i x_i instead of the post-block -- the coefficient tile is `pre` [32 x n]
// (raw, expanded like a post tile) or its lane layout, no y, one output tile per position
constexpr bool HMODE = get_compile_time_arg_val(10) != 0;
static_assert(N + 4 <= 8, "x_i, y, comb, post and out must fit the 8 fp32 dest slots");
static_assert(N % 2 == 0 && N <= 4, "the x8 expansion pairs coefficients and keeps them in the left face");

constexpr uint32_t D_Y = N, D_COMB = N + 1, D_POST = N + 2, D_OUT = N + 3;
constexpr uint32_t COEF_TILES = HMODE ? 1 : N + 1;  // comb lane tiles of streams 0..N-1, then post (or just pre)

// the packer frees the dest for the math thread without zeroing it: the operand slots survive the pack
FORCE_INLINE void release_keep_dest() { PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>())); }

// raw comb + post tiles of the current token-tile (CB_RAW) -> the N + 1 lane-layout tiles of CB_COEF
// (HMODE: the raw pre tile -> one lane-layout tile)
FORCE_INLINE void expand_coefficients() {
    CircularBuffer cbraw(CB_RAW), cbcoef(CB_COEF);
    constexpr uint32_t raw_tiles = HMODE ? 1 : 2;
    cbraw.wait_front(raw_tiles);
    tile_regs_acquire();
    copy_init(CB_RAW);
    copy_tile(CB_RAW, 0, 0);  // comb (or pre)
    if constexpr (!HMODE) {
        copy_tile(CB_RAW, 1, 1);  // post
    }
    sfpu_bcast_col_init();  // SFPU config + zero-stride ADDR_MOD_7 for the hand-written routines
    if constexpr (HMODE) {
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_post, 0, 2, 2, VectorMode::None, N)));
    } else {
        for (uint32_t j = 0; j < N; ++j) {
            MATH((
                _llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_comb, 0, 2 + j, 2 + j, VectorMode::None, N, j)));
        }
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_post, 1, 2 + N, 2 + N, VectorMode::None, N)));
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < COEF_TILES; ++t) {
        cbcoef.reserve_back(1);
        pack_tile(2 + t, CB_COEF);
        cbcoef.push_back(1);
    }
    tile_regs_release();
    cbraw.pop_front(raw_tiles);
}

// HMODE position: h = sum_i pre_i x_i, one output tile
FORCE_INLINE void h_position() {
    CircularBuffer cbx(CB_X), cbcoef(CB_COEF), cbo(CB_OUT);
    cbx.wait_front(N);
    tile_regs_acquire();
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, i, i);
    }
    copy_init(CB_COEF);
    copy_tile(CB_COEF, 0, D_COMB);
    sfpu_bcast_col_init();
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::pre_mix, D_OUT, D_COMB, D_OUT, VectorMode::None, N)));
    tile_regs_commit();
    tile_regs_wait();
    cbo.reserve_back(1);
    pack_tile(D_OUT, CB_OUT);
    cbo.push_back(1);
    tile_regs_release();
    cbx.pop_front(N);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    const uint32_t start_pos = get_arg_val<uint32_t>(0);
    const uint32_t num_pos = get_arg_val<uint32_t>(1);

    CircularBuffer cbx(CB_X), cby(CB_Y), cbcoef(CB_COEF), cbo(CB_OUT);

    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / CT;
        if (r != cur_r) {
            if (cur_r != 0xFFFFFFFFu) {
                cbcoef.pop_front(COEF_TILES);
            }
            if constexpr (EXPAND) {
                expand_coefficients();
            }
            cbcoef.wait_front(COEF_TILES);
            cur_r = r;
        }

        if constexpr (HMODE) {
            h_position();
            continue;
        }
        cbx.wait_front(N);
        cby.wait_front(1);
        tile_regs_acquire();
        copy_init(CB_X);
        for (uint32_t i = 0; i < N; ++i) {
            copy_tile(CB_X, i, i);
        }
        reconfig_data_format_srca(CB_X, CB_Y);
        copy_init(CB_Y);
        copy_tile(CB_Y, 0, D_Y);  // bf16 y widens on unpack (exact)
        reconfig_data_format_srca(CB_Y, CB_COEF);
        copy_init(CB_COEF);
        copy_tile(CB_COEF, N, D_POST);
        sfpu_bcast_col_init();

        for (uint32_t j = 0; j < N; ++j) {
            if (j != 0 && !COLLAPSE) {
                tile_regs_acquire();  // the previous stream was packed and released without clearing
            }
            copy_tile(CB_COEF, j, D_COMB);
            const bool accumulate = COLLAPSE && j != 0;
            const bool scale = COLLAPSE && j == N - 1;
            MATH((_llk_math_eltwise_binary_sfpu_params_(
                mhc_sfpu::stream_mix,
                D_OUT,
                D_COMB,
                D_OUT,
                VectorMode::None,
                D_POST,
                D_Y,
                N,
                j,
                accumulate,
                scale,
                INV_N_BITS)));
            if (COLLAPSE && j != N - 1) {
                continue;
            }
            tile_regs_commit();
            tile_regs_wait();
            cbo.reserve_back(1);
            pack_tile(D_OUT, CB_OUT);
            cbo.push_back(1);
            if (j == N - 1) {
                tile_regs_release();
            } else {
                release_keep_dest();
            }
        }
        cbx.pop_front(N);
        cby.pop_front(1);
    }
    if (cur_r != 0xFFFFFFFFu) {
        cbcoef.pop_front(COEF_TILES);
    }
}
