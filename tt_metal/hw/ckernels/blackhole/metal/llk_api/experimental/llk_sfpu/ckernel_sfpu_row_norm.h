// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-row RMSNorm and LayerNorm SFPU kernels for packed (M, 32) and stacked (16/32, 32) tiles
// (M in {1, 2, 4, 8}, TILE_H in {M, 16, 32}). Faces sit at offset f * 8 SFPU vectors regardless of
// face_r_dim; stacked M in {1, 2} folds subvectors via subvec_transp. Wide-row LayerNorm packs
// mean in STATS_TILE Face 0 and variance/inv_std in Face 1 (hi+lo double-BF16 when !STATS_IS_FP32).
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_row_common.h"
#include "ckernel_sfpu_rsqrt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu::row_norm {

using row_common::kFaceVecs;
using row_common::kTileVecs;
using row_common::round_for_dest;
using row_common::store;
using row_common::subvec_sum;
using row_common::valid_num_rows;

constexpr std::uint32_t kMaxUnrollTiles = 8;

// Hardware full-DEST ceiling; SFPU bodies are sync-mode-agnostic, and the Compute API enforces
// the tighter DST_SYNC_MODE bound (SyncHalf: 4 FP32 / 8 BF16; SyncFull: 8 FP32 / 16 BF16).
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t kMaxFullDestTiles =
    get_dest_max_tiles<DstSync::SyncFull, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H>
constexpr bool valid_tile_h =
    valid_num_rows<NUM_ROWS> &&
    (TILE_H == 1 || TILE_H == 2 || TILE_H == 4 || TILE_H == 8 || TILE_H == 16 || TILE_H == 32) && (TILE_H >= NUM_ROWS);

template <std::uint32_t NUM_TILES>
constexpr bool valid_num_tiles = (NUM_TILES >= 1u && NUM_TILES <= kMaxUnrollTiles);

template <std::uint32_t TILE_H>
constexpr int kFaces = (TILE_H == 32) ? 4 : 2;

template <std::uint32_t TILE_H>
constexpr int kRowGroups = (TILE_H >= 16) ? 4 : (TILE_H == 8 ? 2 : 1);

// Unrolled tile/face/row-group loop drivers keeping dst_reg indices compile-time constant.
template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, typename Fn>
sfpi_inline void for_each_row_group(Fn&& fn) {
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
#pragma GCC unroll 8
    for (std::uint32_t t = 0; t < NUM_TILES; ++t) {
#pragma GCC unroll 4
        for (int f = 0; f < kFaces<TILE_H>; ++f) {
#pragma GCC unroll 4
            for (int g = 0; g < kRowGroups<TILE_H>; ++g) {
                fn(static_cast<int>(t) * kTileVecs + f * kFaceVecs + 2 * g);
            }
        }
    }
}

// Walks even/odd row-group pairs (step 2) for M == 8: v+0..v+1 hold rows 0..3, v+2..v+3 hold rows 4..7.
template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, typename Fn>
sfpi_inline void for_each_row_group_pair(Fn&& fn) {
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
#pragma GCC unroll 8
    for (std::uint32_t t = 0; t < NUM_TILES; ++t) {
#pragma GCC unroll 4
        for (int f = 0; f < kFaces<TILE_H>; ++f) {
#pragma GCC unroll 2
            for (int g = 0; g < kRowGroups<TILE_H>; g += 2) {
                fn(static_cast<int>(t) * kTileVecs + f * kFaceVecs + 2 * g);
            }
        }
    }
}

// Walks one half-band (HALF == 0 for rows 0..3, HALF == 1 for rows 4..7) for single-pass M == 8 LayerNorm.
template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, int HALF, typename Fn>
sfpi_inline void for_each_half_row_group(Fn&& fn) {
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    static_assert(HALF == 0 || HALF == 1, "row_norm: HALF must be 0 or 1");
#pragma GCC unroll 8
    for (std::uint32_t t = 0; t < NUM_TILES; ++t) {
#pragma GCC unroll 4
        for (int f = 0; f < kFaces<TILE_H>; ++f) {
#pragma GCC unroll 2
            for (int g = HALF; g < kRowGroups<TILE_H>; g += 2) {
                fn(static_cast<int>(t) * kTileVecs + f * kFaceVecs + 2 * g);
            }
        }
    }
}

// Horizontal column reduction + subvector folding whenever TILE_H > NUM_ROWS for M in {1, 2}.
template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H>
sfpi_inline sfpi::vFloat reduce_row_lanes(sfpi::vFloat v) {
    v = subvec_sum(v);
    if constexpr (NUM_ROWS == 1 && TILE_H == 2) {
        sfpi::vFloat a = v, b = v, c = v, d = v;
        sfpi::vFloat z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
        sfpi::subvec_transp(a, b, c, d, z0, z1, z2, z3);
        a = a + b;
        return a;
    } else if constexpr (NUM_ROWS == 1 && TILE_H >= 4) {
        sfpi::vFloat a = v, b = v, c = v, d = v;
        sfpi::vFloat z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
        sfpi::subvec_transp(a, b, c, d, z0, z1, z2, z3);
        a = (a + b) + (c + d);
        return a;
    } else if constexpr (NUM_ROWS == 2 && TILE_H >= 4) {
        sfpi::vFloat a = v, b = v, c = v, d = v;
        sfpi::vFloat z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
        sfpi::subvec_transp(a, b, c, d, z0, z1, z2, z3);
        a = a + c;
        b = b + d;
        c = a;
        d = b;
        sfpi::subvec_transp(a, b, c, d, z0, z1, z2, z3);
        return a;
    } else {
        return v;
    }
}

// Compact per-row statistic load/store inside one face (FACE in {0, 1}) of TILE.
// When STATS_IS_FP32 is false (BF16 DEST or a BF16 L1 stats CB), stores hi + lo double-BF16 across
// even/odd column vectors of the same row-group (~15-bit mantissa preserved across chunks and through
// BF16 L1 pack/copy_tile for every M in {1, 2, 4, 8}, even when DEST itself runs in FP32).
template <std::uint32_t TILE, int FACE, std::uint32_t NUM_ROWS, bool STATS_IS_FP32>
sfpi_inline void load_packed_stat(sfpi::vFloat& top, sfpi::vFloat& bot) {
    constexpr int base = static_cast<int>(TILE) * kTileVecs + FACE * kFaceVecs;
    if constexpr (STATS_IS_FP32) {
        top = sfpi::dst_reg[base + 0];
        if constexpr (NUM_ROWS == 8) {
            bot = sfpi::dst_reg[base + 2];
        }
    } else {
        sfpi::vFloat hi = sfpi::dst_reg[base + 0];
        sfpi::vFloat lo = sfpi::dst_reg[base + 1];
        top = hi + lo;
        if constexpr (NUM_ROWS == 8) {
            sfpi::vFloat bhi = sfpi::dst_reg[base + 2];
            sfpi::vFloat blo = sfpi::dst_reg[base + 3];
            bot = bhi + blo;
        }
    }
}

template <std::uint32_t TILE, int FACE, std::uint32_t NUM_ROWS, bool STATS_IS_FP32>
sfpi_inline void store_packed_stat(sfpi::vFloat top, sfpi::vFloat bot) {
    constexpr int base = static_cast<int>(TILE) * kTileVecs + FACE * kFaceVecs;
    if constexpr (STATS_IS_FP32) {
        sfpi::dst_reg[base + 0] = top;
        if constexpr (NUM_ROWS == 8) {
            sfpi::dst_reg[base + 2] = bot;
        }
    } else {
        sfpi::vFloat hi = round_for_dest<false>(top);
        sfpi::vFloat lo = round_for_dest<false>(top - hi);
        sfpi::dst_reg[base + 0] = hi;
        sfpi::dst_reg[base + 1] = lo;
        if constexpr (NUM_ROWS == 8) {
            sfpi::vFloat bhi = round_for_dest<false>(bot);
            sfpi::vFloat blo = round_for_dest<false>(bot - bhi);
            sfpi::dst_reg[base + 2] = bhi;
            sfpi::dst_reg[base + 3] = blo;
        }
    }
}

// Replicate a row vector across the footprint of a ROW_H-row tile in slot TILE so FPU
// bcast_dest_reuse (face 0, BroadcastType::COL) or mul_tiles_bcast_cols (all faces) can consume column 0 directly.
template <
    std::uint32_t TILE,
    std::uint32_t NUM_ROWS,
    bool is_fp32_dest_acc_en,
    std::uint32_t ROW_H = NUM_ROWS,
    bool FACE0_ONLY = false>
sfpi_inline void store_row_vector(sfpi::vFloat top, sfpi::vFloat bot) {
    static_assert(valid_tile_h<NUM_ROWS, ROW_H>, "row_norm: invalid (NUM_ROWS, ROW_H)");
    static_assert(TILE < kMaxFullDestTiles<is_fp32_dest_acc_en>, "row_norm: TILE exceeds DEST capacity");
    constexpr int nfaces = FACE0_ONLY ? 1 : kFaces<ROW_H>;
    if constexpr (NUM_ROWS == 8) {
#pragma GCC unroll 4
        for (int f = 0; f < nfaces; ++f) {
#pragma GCC unroll 2
            for (int g = 0; g < kRowGroups<ROW_H>; g += 2) {
                const int v = static_cast<int>(TILE) * kTileVecs + f * kFaceVecs + 2 * g;
                store<is_fp32_dest_acc_en>(v + 0, top);
                store<is_fp32_dest_acc_en>(v + 1, top);
                store<is_fp32_dest_acc_en>(v + 2, bot);
                store<is_fp32_dest_acc_en>(v + 3, bot);
            }
        }
    } else {
        (void)bot;
#pragma GCC unroll 4
        for (int f = 0; f < nfaces; ++f) {
#pragma GCC unroll 4
            for (int g = 0; g < kRowGroups<ROW_H>; ++g) {
                const int v = static_cast<int>(TILE) * kTileVecs + f * kFaceVecs + 2 * g;
                store<is_fp32_dest_acc_en>(v + 0, top);
                store<is_fp32_dest_acc_en>(v + 1, top);
            }
        }
    }
}

// Half-band helpers for single-pass M == 8 LayerNorm: processing rows 0..3 (HALF=0) then
// rows 4..7 (HALF=1) keeps peak live LRegs across rsqrt at 2 (mean + var) instead of 4.
template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, int HALF>
sfpi_inline sfpi::vFloat sum_values_half() {
    sfpi::vFloat acc0 = 0.0f;
    sfpi::vFloat acc1 = 0.0f;
    for_each_half_row_group<TILE_H, NUM_TILES, HALF>([&](int v) __attribute__((always_inline)) {
        acc0 += sfpi::dst_reg[v + 0];
        acc1 += sfpi::dst_reg[v + 1];
    });
    return subvec_sum(acc0 + acc1);
}

template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, int HALF>
sfpi_inline sfpi::vFloat sum_centered_squares_half(sfpi::vFloat mean) {
    sfpi::vFloat acc0 = 0.0f;
    sfpi::vFloat acc1 = 0.0f;
    for_each_half_row_group<TILE_H, NUM_TILES, HALF>([&](int v) __attribute__((always_inline)) {
        sfpi::vFloat d0 = sfpi::dst_reg[v + 0] - mean;
        sfpi::vFloat d1 = sfpi::dst_reg[v + 1] - mean;
        acc0 += d0 * d0;
        acc1 += d1 * d1;
    });
    return subvec_sum(acc0 + acc1);
}

template <std::uint32_t TILE_H, std::uint32_t NUM_TILES, int HALF, bool is_fp32_dest_acc_en>
sfpi_inline void center_scale_half(sfpi::vFloat mean, sfpi::vFloat inv_std) {
    if constexpr (!is_fp32_dest_acc_en) {
        const sfpi::vFloat neg_mean_inv = -(mean * inv_std);
        for_each_half_row_group<TILE_H, NUM_TILES, HALF>([&](int v) __attribute__((always_inline)) {
            store<false>(v + 0, sfpi::dst_reg[v + 0] * inv_std + neg_mean_inv);
            store<false>(v + 1, sfpi::dst_reg[v + 1] * inv_std + neg_mean_inv);
        });
    } else {
        for_each_half_row_group<TILE_H, NUM_TILES, HALF>([&](int v) __attribute__((always_inline)) {
            store<true>(v + 0, (sfpi::dst_reg[v + 0] - mean) * inv_std);
            store<true>(v + 1, (sfpi::dst_reg[v + 1] - mean) * inv_std);
        });
    }
}

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H, std::uint32_t NUM_TILES>
sfpi_inline void sum_values(sfpi::vFloat& top, sfpi::vFloat& bot) {
    static_assert(valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (NUM_ROWS == 8) {
        sfpi::vFloat acc_top = 0.0f;
        sfpi::vFloat acc_bot = 0.0f;
        for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            acc_top += sfpi::dst_reg[v + 0];
            acc_bot += sfpi::dst_reg[v + 2];
            acc_top += sfpi::dst_reg[v + 1];
            acc_bot += sfpi::dst_reg[v + 3];
        });
        top = subvec_sum(acc_top);
        bot = subvec_sum(acc_bot);
    } else {
        sfpi::vFloat acc0 = 0.0f;
        sfpi::vFloat acc1 = 0.0f;
        for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            acc0 += sfpi::dst_reg[v + 0];
            acc1 += sfpi::dst_reg[v + 1];
        });
        top = reduce_row_lanes<NUM_ROWS, TILE_H>(acc0 + acc1);
    }
}

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H, std::uint32_t NUM_TILES>
sfpi_inline void sum_squares(sfpi::vFloat& top, sfpi::vFloat& bot) {
    static_assert(valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (NUM_ROWS == 8) {
        sfpi::vFloat acc_top = 0.0f;
        sfpi::vFloat acc_bot = 0.0f;
        for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            sfpi::vFloat x0 = sfpi::dst_reg[v + 0];
            sfpi::vFloat x2 = sfpi::dst_reg[v + 2];
            acc_top += x0 * x0;
            acc_bot += x2 * x2;
            sfpi::vFloat x1 = sfpi::dst_reg[v + 1];
            sfpi::vFloat x3 = sfpi::dst_reg[v + 3];
            acc_top += x1 * x1;
            acc_bot += x3 * x3;
        });
        top = subvec_sum(acc_top);
        bot = subvec_sum(acc_bot);
    } else {
        sfpi::vFloat acc0 = 0.0f;
        sfpi::vFloat acc1 = 0.0f;
        for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            sfpi::vFloat x0 = sfpi::dst_reg[v + 0];
            sfpi::vFloat x1 = sfpi::dst_reg[v + 1];
            acc0 += x0 * x0;
            acc1 += x1 * x1;
        });
        top = reduce_row_lanes<NUM_ROWS, TILE_H>(acc0 + acc1);
    }
}

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H, std::uint32_t NUM_TILES>
sfpi_inline void sum_centered_squares(
    sfpi::vFloat mean_top, sfpi::vFloat mean_bot, sfpi::vFloat& var_top, sfpi::vFloat& var_bot) {
    static_assert(valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (NUM_ROWS == 8) {
        sfpi::vFloat acc_top = 0.0f;
        sfpi::vFloat acc_bot = 0.0f;
        for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            sfpi::vFloat d0 = sfpi::dst_reg[v + 0] - mean_top;
            sfpi::vFloat d2 = sfpi::dst_reg[v + 2] - mean_bot;
            acc_top += d0 * d0;
            acc_bot += d2 * d2;
            sfpi::vFloat d1 = sfpi::dst_reg[v + 1] - mean_top;
            sfpi::vFloat d3 = sfpi::dst_reg[v + 3] - mean_bot;
            acc_top += d1 * d1;
            acc_bot += d3 * d3;
        });
        var_top = subvec_sum(acc_top);
        var_bot = subvec_sum(acc_bot);
    } else {
        (void)mean_bot;
        sfpi::vFloat acc0 = 0.0f;
        sfpi::vFloat acc1 = 0.0f;
        for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            sfpi::vFloat d0 = sfpi::dst_reg[v + 0] - mean_top;
            sfpi::vFloat d1 = sfpi::dst_reg[v + 1] - mean_top;
            acc0 += d0 * d0;
            acc1 += d1 * d1;
        });
        var_top = reduce_row_lanes<NUM_ROWS, TILE_H>(acc0 + acc1);
    }
}

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H, std::uint32_t NUM_TILES, bool is_fp32_dest_acc_en>
sfpi_inline void scale_tiles(sfpi::vFloat top, sfpi::vFloat bot) {
    static_assert(valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (NUM_ROWS == 8) {
        for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            store<is_fp32_dest_acc_en>(v + 0, sfpi::dst_reg[v + 0] * top);
            store<is_fp32_dest_acc_en>(v + 1, sfpi::dst_reg[v + 1] * top);
            store<is_fp32_dest_acc_en>(v + 2, sfpi::dst_reg[v + 2] * bot);
            store<is_fp32_dest_acc_en>(v + 3, sfpi::dst_reg[v + 3] * bot);
        });
    } else {
        (void)bot;
        for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            store<is_fp32_dest_acc_en>(v + 0, sfpi::dst_reg[v + 0] * top);
            store<is_fp32_dest_acc_en>(v + 1, sfpi::dst_reg[v + 1] * top);
        });
    }
}

template <std::uint32_t NUM_ROWS, std::uint32_t TILE_H, std::uint32_t NUM_TILES, bool is_fp32_dest_acc_en>
sfpi_inline void center_scale_tiles(
    sfpi::vFloat mean_top, sfpi::vFloat mean_bot, sfpi::vFloat inv_top, sfpi::vFloat inv_bot) {
    static_assert(valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (!is_fp32_dest_acc_en) {
        const sfpi::vFloat neg_top = -(mean_top * inv_top);
        if constexpr (NUM_ROWS == 8) {
            const sfpi::vFloat neg_bot = -(mean_bot * inv_bot);
            for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
                store<false>(v + 0, sfpi::dst_reg[v + 0] * inv_top + neg_top);
                store<false>(v + 1, sfpi::dst_reg[v + 1] * inv_top + neg_top);
                store<false>(v + 2, sfpi::dst_reg[v + 2] * inv_bot + neg_bot);
                store<false>(v + 3, sfpi::dst_reg[v + 3] * inv_bot + neg_bot);
            });
        } else {
            (void)mean_bot;
            (void)inv_bot;
            for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
                store<false>(v + 0, sfpi::dst_reg[v + 0] * inv_top + neg_top);
                store<false>(v + 1, sfpi::dst_reg[v + 1] * inv_top + neg_top);
            });
        }
    } else if constexpr (NUM_ROWS == 8) {
        for_each_row_group_pair<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            store<true>(v + 0, (sfpi::dst_reg[v + 0] - mean_top) * inv_top);
            store<true>(v + 1, (sfpi::dst_reg[v + 1] - mean_top) * inv_top);
            store<true>(v + 2, (sfpi::dst_reg[v + 2] - mean_bot) * inv_bot);
            store<true>(v + 3, (sfpi::dst_reg[v + 3] - mean_bot) * inv_bot);
        });
    } else {
        (void)mean_bot;
        (void)inv_bot;
        for_each_row_group<TILE_H, NUM_TILES>([&](int v) __attribute__((always_inline)) {
            store<true>(v + 0, (sfpi::dst_reg[v + 0] - mean_top) * inv_top);
            store<true>(v + 1, (sfpi::dst_reg[v + 1] - mean_top) * inv_top);
        });
    }
}

// Compute rsqrt(v * inv_width + eps) keeping constants dead before _calculate_sqrt_body_.
template <std::uint32_t NUM_ROWS, bool APPROXIMATION_MODE, bool FAST_APPROX, bool SCALE_SUM>
sfpi_inline void scale_add_rsqrt(
    sfpi::vFloat& top, sfpi::vFloat& bot, const std::uint32_t inv_width_bits, const std::uint32_t epsilon_bits) {
    {
        const sfpi::vFloat eps = Converter::as_float(epsilon_bits);
        if constexpr (SCALE_SUM) {
            const sfpi::vFloat inv_w = Converter::as_float(inv_width_bits);
            top = top * inv_w + eps;
            if constexpr (NUM_ROWS == 8) {
                bot = bot * inv_w + eps;
            }
        } else {
            (void)inv_width_bits;
            top = top + eps;
            if constexpr (NUM_ROWS == 8) {
                bot = bot + eps;
            }
        }
    }
    top = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(top);
    if constexpr (NUM_ROWS == 8) {
        bot = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(bot);
    }
}

}  // namespace ckernel::sfpu::row_norm

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void _row_norm_init_() {
    sqrt_init<APPROXIMATION_MODE>();
}

// Entry points are noinline to scope GCC's rvtt_replay buffer per leaf, and end with WAIT_SFPU
// so follow-on FPU ops see completed DEST writes.

// Single-pass in-DST per-row RMSNorm across NUM_TILES tiles starting at dst_reg[0].
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    bool SCALE_SUM = true>
__attribute__((noinline)) inline void _calculate_row_rmsnorm_(
    const std::uint32_t inv_width_bits, const std::uint32_t epsilon_bits) {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    sfpi::vFloat top, bot;
    row_norm::sum_squares<NUM_ROWS, TILE_H, NUM_TILES>(top, bot);
    row_norm::scale_add_rsqrt<NUM_ROWS, APPROXIMATION_MODE, FAST_APPROX, SCALE_SUM>(
        top, bot, inv_width_bits, epsilon_bits);
    row_norm::scale_tiles<NUM_ROWS, TILE_H, NUM_TILES, is_fp32_dest_acc_en>(top, bot);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// FPU-assisted RMSNorm tail: sums NUM_TILES pre-squared tiles [0, NUM_TILES) in FP32, computes
// inv_rms = rsqrt(sum * inv_width + eps), and replicates inv_rms into tile 0 (face 0 only when
// FACE0_ONLY is true for bcast_dest_reuse BroadcastType::COL, or all faces for L1 pack + mul_tiles_bcast_cols).
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    bool SCALE_SUM = true,
    bool FACE0_ONLY = false>
__attribute__((noinline)) inline void _calculate_row_reduce_scale_add_rsqrt_(
    const std::uint32_t inv_width_bits, const std::uint32_t epsilon_bits) {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    sfpi::vFloat top, bot;
    row_norm::sum_values<NUM_ROWS, TILE_H, NUM_TILES>(top, bot);
    row_norm::scale_add_rsqrt<NUM_ROWS, APPROXIMATION_MODE, FAST_APPROX, SCALE_SUM>(
        top, bot, inv_width_bits, epsilon_bits);
    row_norm::store_row_vector<0, NUM_ROWS, is_fp32_dest_acc_en, TILE_H, FACE0_ONLY>(top, bot);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Single-pass in-DST per-row LayerNorm across NUM_TILES tiles starting at dst_reg[0].
// Keeps mean, centered differences, and variance in FP32 registers (no BF16 centered scratch).
// For stacked M in {1, 2} (TILE_H in {16, 32}), SCRATCH_TILE (default NUM_TILES) holds mean across
// subvec_transp (using hi+lo double-BF16 when !is_fp32_dest_acc_en to preserve ~15-bit mantissa).
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    std::uint32_t SCRATCH_TILE = NUM_TILES>
__attribute__((noinline)) inline void _calculate_row_layernorm_(
    const std::uint32_t inv_width_bits, const std::uint32_t epsilon_bits) {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    if constexpr (NUM_ROWS == 8) {
        {
            sfpi::vFloat mean = row_norm::sum_values_half<TILE_H, NUM_TILES, 0>();
            mean *= Converter::as_float(inv_width_bits);
            sfpi::vFloat var = row_norm::sum_centered_squares_half<TILE_H, NUM_TILES, 0>(mean);
            var = var * Converter::as_float(inv_width_bits) + Converter::as_float(epsilon_bits);
            sfpi::vFloat inv_std = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(var);
            row_norm::center_scale_half<TILE_H, NUM_TILES, 0, is_fp32_dest_acc_en>(mean, inv_std);
        }
        {
            sfpi::vFloat mean = row_norm::sum_values_half<TILE_H, NUM_TILES, 1>();
            mean *= Converter::as_float(inv_width_bits);
            sfpi::vFloat var = row_norm::sum_centered_squares_half<TILE_H, NUM_TILES, 1>(mean);
            var = var * Converter::as_float(inv_width_bits) + Converter::as_float(epsilon_bits);
            sfpi::vFloat inv_std = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(var);
            row_norm::center_scale_half<TILE_H, NUM_TILES, 1, is_fp32_dest_acc_en>(mean, inv_std);
        }
    } else if constexpr (NUM_ROWS <= 2 && TILE_H > NUM_ROWS) {
        static_assert(
            SCRATCH_TILE >= NUM_TILES && SCRATCH_TILE < row_norm::kMaxFullDestTiles<is_fp32_dest_acc_en>,
            "row_norm: SCRATCH_TILE must lie in [NUM_TILES, DEST capacity)");
        // subvec_transp clobbers all LRegs on Blackhole, so stash mean in SCRATCH_TILE across it.
        sfpi::vFloat var;
        {
            sfpi::vFloat mean, unused;
            row_norm::sum_values<NUM_ROWS, TILE_H, NUM_TILES>(mean, unused);
            mean *= Converter::as_float(inv_width_bits);
            row_norm::store_packed_stat<SCRATCH_TILE, 0, NUM_ROWS, is_fp32_dest_acc_en>(mean, unused);
            row_norm::sum_centered_squares<NUM_ROWS, TILE_H, NUM_TILES>(mean, unused, var, unused);
        }
        var = var * Converter::as_float(inv_width_bits) + Converter::as_float(epsilon_bits);
        sfpi::vFloat inv_std = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(var);
        sfpi::vFloat mean, unused;
        row_norm::load_packed_stat<SCRATCH_TILE, 0, NUM_ROWS, is_fp32_dest_acc_en>(mean, unused);
        row_norm::center_scale_tiles<NUM_ROWS, TILE_H, NUM_TILES, is_fp32_dest_acc_en>(mean, unused, inv_std, unused);
    } else {
        sfpi::vFloat mean, unused;
        row_norm::sum_values<NUM_ROWS, TILE_H, NUM_TILES>(mean, unused);
        mean *= Converter::as_float(inv_width_bits);

        sfpi::vFloat var;
        row_norm::sum_centered_squares<NUM_ROWS, TILE_H, NUM_TILES>(mean, unused, var, unused);
        var = var * Converter::as_float(inv_width_bits) + Converter::as_float(epsilon_bits);
        sfpi::vFloat inv_std = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(var);
        row_norm::center_scale_tiles<NUM_ROWS, TILE_H, NUM_TILES, is_fp32_dest_acc_en>(mean, unused, inv_std, unused);
    }
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Wide-row LayerNorm Pass 1a tail: sums NUM_TILES FPU-accumulated tiles [0, NUM_TILES), scales by
// inv_width, and stores mean compactly in Face 0 of STATS_TILE.
template <
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    std::uint32_t STATS_TILE,
    bool STATS_IS_FP32 = is_fp32_dest_acc_en>
__attribute__((noinline)) inline void _calculate_row_reduce_mean_(const std::uint32_t inv_width_bits) {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    static_assert(
        STATS_TILE >= NUM_TILES && STATS_TILE < row_norm::kMaxFullDestTiles<is_fp32_dest_acc_en>,
        "row_norm: STATS_TILE must lie in [NUM_TILES, DEST capacity)");
    sfpi::vFloat top, bot;
    row_norm::sum_values<NUM_ROWS, TILE_H, NUM_TILES>(top, bot);
    const sfpi::vFloat inv_w = Converter::as_float(inv_width_bits);
    top *= inv_w;
    if constexpr (NUM_ROWS == 8) {
        bot *= inv_w;
    }
    row_norm::store_packed_stat<STATS_TILE, 0, NUM_ROWS, STATS_IS_FP32>(top, bot);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Wide-row LayerNorm Pass 1b chunk: reads mean from Face 0 of STATS_TILE (unless ZERO_MEAN is
// true) and accumulates sum((x - mean)^2) over tiles [0, NUM_TILES) into Face 1 of STATS_TILE.
template <
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    std::uint32_t STATS_TILE,
    bool ACCUMULATE,
    bool ZERO_MEAN = false,
    bool STATS_IS_FP32 = is_fp32_dest_acc_en>
__attribute__((noinline)) inline void _calculate_row_sum_centered_squares_() {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    static_assert(
        STATS_TILE >= NUM_TILES && STATS_TILE < row_norm::kMaxFullDestTiles<is_fp32_dest_acc_en>,
        "row_norm: STATS_TILE must lie in [NUM_TILES, DEST capacity)");
    sfpi::vFloat var_top, var_bot;
    if constexpr (ZERO_MEAN) {
        row_norm::sum_squares<NUM_ROWS, TILE_H, NUM_TILES>(var_top, var_bot);
    } else {
        sfpi::vFloat mean_top, mean_bot;
        row_norm::load_packed_stat<STATS_TILE, 0, NUM_ROWS, STATS_IS_FP32>(mean_top, mean_bot);
        row_norm::sum_centered_squares<NUM_ROWS, TILE_H, NUM_TILES>(mean_top, mean_bot, var_top, var_bot);
    }
    if constexpr (ACCUMULATE) {
        sfpi::vFloat prev_top, prev_bot;
        row_norm::load_packed_stat<STATS_TILE, 1, NUM_ROWS, STATS_IS_FP32>(prev_top, prev_bot);
        var_top += prev_top;
        if constexpr (NUM_ROWS == 8) {
            var_bot += prev_bot;
        }
    }
    row_norm::store_packed_stat<STATS_TILE, 1, NUM_ROWS, STATS_IS_FP32>(var_top, var_bot);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Wide-row LayerNorm Pass 1b finalizer: converts the accumulated sum of squared deviations in
// Face 1 of STATS_TILE into inv_std = rsqrt(sum * inv_width + eps) in place.
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t STATS_TILE,
    bool STATS_IS_FP32 = is_fp32_dest_acc_en>
__attribute__((noinline)) inline void _calculate_row_finalize_inv_std_(
    const std::uint32_t inv_width_bits, const std::uint32_t epsilon_bits) {
    static_assert(row_norm::valid_num_rows<NUM_ROWS>, "row_norm: NUM_ROWS must be 1, 2, 4, or 8");
    static_assert(
        STATS_TILE < row_norm::kMaxFullDestTiles<is_fp32_dest_acc_en>, "row_norm: STATS_TILE exceeds DEST capacity");
    sfpi::vFloat var_top, var_bot;
    row_norm::load_packed_stat<STATS_TILE, 1, NUM_ROWS, STATS_IS_FP32>(var_top, var_bot);
    row_norm::scale_add_rsqrt<NUM_ROWS, APPROXIMATION_MODE, FAST_APPROX, true>(
        var_top, var_bot, inv_width_bits, epsilon_bits);
    row_norm::store_packed_stat<STATS_TILE, 1, NUM_ROWS, STATS_IS_FP32>(var_top, var_bot);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Wide-row LayerNorm Pass 2 chunk normalize: reads mean (Face 0, unless ZERO_MEAN is true) and
// inv_std (Face 1) from STATS_TILE and applies (x - mean) * inv_std across tiles [0, NUM_TILES).
template <
    bool is_fp32_dest_acc_en,
    std::uint32_t NUM_ROWS,
    std::uint32_t TILE_H,
    std::uint32_t NUM_TILES,
    std::uint32_t STATS_TILE,
    bool ZERO_MEAN = false,
    bool STATS_IS_FP32 = is_fp32_dest_acc_en>
__attribute__((noinline)) inline void _calculate_row_center_scale_() {
    static_assert(row_norm::valid_tile_h<NUM_ROWS, TILE_H>, "row_norm: invalid (NUM_ROWS, TILE_H)");
    static_assert(row_norm::valid_num_tiles<NUM_TILES>, "row_norm: NUM_TILES must be in [1, 8]");
    static_assert(
        STATS_TILE >= NUM_TILES && STATS_TILE < row_norm::kMaxFullDestTiles<is_fp32_dest_acc_en>,
        "row_norm: STATS_TILE must lie in [NUM_TILES, DEST capacity)");
    sfpi::vFloat inv_top, inv_bot;
    row_norm::load_packed_stat<STATS_TILE, 1, NUM_ROWS, STATS_IS_FP32>(inv_top, inv_bot);
    if constexpr (ZERO_MEAN) {
        row_norm::scale_tiles<NUM_ROWS, TILE_H, NUM_TILES, is_fp32_dest_acc_en>(inv_top, inv_bot);
    } else {
        sfpi::vFloat mean_top, mean_bot;
        row_norm::load_packed_stat<STATS_TILE, 0, NUM_ROWS, STATS_IS_FP32>(mean_top, mean_bot);
        row_norm::center_scale_tiles<NUM_ROWS, TILE_H, NUM_TILES, is_fp32_dest_acc_en>(
            mean_top, mean_bot, inv_top, inv_bot);
    }
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

}  // namespace ckernel::sfpu
