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
