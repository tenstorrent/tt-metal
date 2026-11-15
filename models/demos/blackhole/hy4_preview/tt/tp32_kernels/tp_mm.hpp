// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// skinny-matmul chains, TRISC helpers: the blaze streaming matmul (custom_mm, tiny 8x32 in0 tiles, LoFi, fp32 dest
// accumulation) generalised to
//   * MT row tiles of in0 (T = 9..16 rows as two 8x32 tiles; the same weight block is multiplied with every row tile),
//   * an in0 tile offset (K-sliced phases),
// out layout per sub-block of SBW columns: [row tile 0: SBW tiles][row tile 1: SBW tiles]...
// in0 CB holds MT groups of KT tiles: row tile m at tile index m * KT.

#pragma once
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/pack_block.h"
#include "api/dataflow/circular_buffer.h"

template <
    uint32_t CB_A,
    uint32_t CB_W,
    uint32_t CB_O,
    uint32_t SBK,
    uint32_t PN,
    uint32_t SBW,
    uint32_t NSUBK,
    uint32_t MT,
    bool FP32,
    uint32_t CT = 1>
FORCE_INLINE void sk_mm(uint32_t a_off, uint32_t kt_stride) {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O, CT);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, CT);
    }
    pack_block_contiguous_init(CB_O);
    static_assert(
        PN % SBW == 0 && SBW % CT == 0,
        "per-core columns must be a multiple of the sub-block width, which a multiple of ct_dim");
    static_assert(MT * SBW <= (FP32 ? 4 : 8), "dest registers");
    for (uint32_t sb_n = 0; sb_n < PN / SBW; ++sb_n) {
        cb_reserve_back(CB_O, MT * SBW);
        tile_regs_acquire();
        for (uint32_t w = 0; w < SBW; w += CT) {
            for (uint32_t sb_k = 0; sb_k < NSUBK; ++sb_k) {
                cb_wait_front(CB_W, SBK * CT);
                if (sb_k + 1 < NSUBK) {
                    for (uint32_t m = 0; m < MT; ++m) {
                        custom_mm_block<false>(CB_A, CB_W, a_off + m * kt_stride + sb_k * SBK, 0, m * SBW + w, SBK, CT);
                    }
                } else {
                    for (uint32_t m = 0; m < MT; ++m) {
                        custom_mm_block<true>(CB_A, CB_W, a_off + m * kt_stride + sb_k * SBK, 0, m * SBW + w, SBK, CT);
                    }
                }
                cb_pop_front(CB_W, SBK * CT);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, MT * SBW);
        tile_regs_release();
        cb_push_back(CB_O, MT * SBW);
    }
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
}

// sk_mm with an epilogue on the accumulators: after the K loops of a sub-block (MT * SBW tiles in dest, tile m * SBW +
// w) `epi()` runs on dest, then NPACK tiles (dest tiles 0 .. NPACK - 1) are packed to CB_O.  Used by the shared expert:
// dest[0] = gate, dest[1] = up -> act = silu(min(g, L)) * clamp(u, -L, L) in fp32, ONE tile packed.
template <
    uint32_t CB_A,
    uint32_t CB_W,
    uint32_t CB_O,
    uint32_t SBK,
    uint32_t PN,
    uint32_t SBW,
    uint32_t NSUBK,
    uint32_t MT,
    bool FP32,
    uint32_t NPACK,
    typename EPI>
FORCE_INLINE void sk_mm_epi(uint32_t a_off, uint32_t kt_stride, EPI epi) {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O, 1);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, 1);
    }
    pack_block_contiguous_init(CB_O);
    static_assert(PN % SBW == 0, "per-core columns must be a multiple of the sub-block width");
    static_assert(MT * SBW <= (FP32 ? 4 : 8), "dest registers");
    for (uint32_t sb_n = 0; sb_n < PN / SBW; ++sb_n) {
        cb_reserve_back(CB_O, NPACK);
        tile_regs_acquire();
        for (uint32_t w = 0; w < SBW; ++w) {
            for (uint32_t sb_k = 0; sb_k < NSUBK; ++sb_k) {
                cb_wait_front(CB_W, SBK);
                if (sb_k + 1 < NSUBK) {
                    for (uint32_t m = 0; m < MT; ++m) {
                        custom_mm_block<false>(CB_A, CB_W, a_off + m * kt_stride + sb_k * SBK, 0, m * SBW + w, SBK, 1);
                    }
                } else {
                    for (uint32_t m = 0; m < MT; ++m) {
                        custom_mm_block<true>(CB_A, CB_W, a_off + m * kt_stride + sb_k * SBK, 0, m * SBW + w, SBK, 1);
                    }
                }
                cb_pop_front(CB_W, SBK);
            }
        }
        epi();
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, NPACK);
        tile_regs_release();
        cb_push_back(CB_O, NPACK);
    }
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
}

// ---- X40 TP32 routed experts: down projection accumulated over ALL active experts in DEST
// -------------------------------------------------------------- cb_a: act tiles, expert a at tiles [2a, 2a + 2) (the
// two K tiles of the 64-wide intermediate slice); cb_w: per (sub-block s, expert a) one block of 8 tiles = [k 2][n 4]
// (CT = 4); out: per sub-block s (6) 4 tiles (hidden tiles 4s .. 4s+3 of this bank's 24) summed over the na active
// experts.  MT = 1 (R <= 8 rows), fp32 dest (4 tiles).
template <uint32_t CB_A, uint32_t CB_W, uint32_t CB_O, uint32_t NSB, bool FP32>
FORCE_INLINE void tp_down(uint32_t na) {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O, 4);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, 4);
    }
    pack_block_contiguous_init(CB_O);
    for (uint32_t s = 0; s < NSB; ++s) {
        cb_reserve_back(CB_O, 4);
        tile_regs_acquire();
        for (uint32_t a = 0; a < na; ++a) {
            cb_wait_front(CB_W, 8);
            if (a + 1 < na) {
                custom_mm_block<false>(CB_A, CB_W, 2 * a, 0, 0, 2, 4);
            } else {
                custom_mm_block<true>(CB_A, CB_W, 2 * a, 0, 0, 2, 4);
            }
            cb_pop_front(CB_W, 8);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, 4);
        tile_regs_release();
        cb_push_back(CB_O, 4);
    }
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
}

// ---- T = 9..16 (MT = 2 row tiles of 8 rows): the gate/up block of an expert (96 tiles = 24 k x [g0 g1 u0 u1]) stays
// in cb_w while MT sequential passes (one fp32 dest set of 4 tiles each) multiply it with row tile m (in0 tiles m * 24
// ..); the weight bytes are read from DRAM ONCE.  Out: MT x 4 tiles per expert, [m][g0 g1 u0 u1].
template <uint32_t CB_A, uint32_t CB_W, uint32_t CB_O, uint32_t MT, bool FP32>
FORCE_INLINE void tp_gu() {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O, 4);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, 4);
    }
    pack_block_contiguous_init(CB_O);
    cb_wait_front(CB_W, 96);
    for (uint32_t m = 0; m < MT; ++m) {
        cb_reserve_back(CB_O, 4);
        tile_regs_acquire();
        custom_mm_block<true>(CB_A, CB_W, m * 24, 0, 0, 24, 4);
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, 4);
        tile_regs_release();
        cb_push_back(CB_O, 4);
    }
    cb_pop_front(CB_W, 96);
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
}

// down for MT row tiles: DEST holds MT x 4 tiles (bf16 dest when MT * 4 > 4, i.e. FP32 = false; 8 tiles), accumulated
// over the active experts.  cb_a: expert a, row tile m, K tile k at tile a * 2 * MT + m * 2 + k.  Out per sub-block:
// [m][n 4].
template <uint32_t CB_A, uint32_t CB_W, uint32_t CB_O, uint32_t NSB, uint32_t MT, bool FP32>
FORCE_INLINE void tp_down_mt(uint32_t na) {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    static_assert(MT * 4 <= (FP32 ? 4 : 8), "dest registers");
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O, 4);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, 4);
    }
    pack_block_contiguous_init(CB_O);
    for (uint32_t s = 0; s < NSB; ++s) {
        cb_reserve_back(CB_O, 4 * MT);
        tile_regs_acquire();
        for (uint32_t a = 0; a < na; ++a) {
            cb_wait_front(CB_W, 8);
            for (uint32_t m = 0; m < MT; ++m) {
                if (a + 1 < na) {
                    custom_mm_block<false>(CB_A, CB_W, a * 2 * MT + m * 2, 0, m * 4, 2, 4);
                } else {
                    custom_mm_block<true>(CB_A, CB_W, a * 2 * MT + m * 2, 0, m * 4, 2, 4);
                }
            }
            cb_pop_front(CB_W, 8);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous<FP32>(0, CB_O, 4 * MT);
        tile_regs_release();
        cb_push_back(CB_O, 4 * MT);
    }
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
}
