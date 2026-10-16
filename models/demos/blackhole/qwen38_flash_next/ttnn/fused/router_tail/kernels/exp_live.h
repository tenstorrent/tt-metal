// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail: the precise exp over the live SFPU vectors of a tile only.
//
// `exp_tile<false>(idst)` (api/compute/eltwise_unary/exp.h) runs `calculate_exponential<false, fp32_dest>` under
// VectorMode::RC: four faces, and per face eight SFPU vectors of 32 datums, each `dst_reg[0] = exp(dst_reg[0]);
// dst_reg++` (the DST address advances by 2).  Blackhole's SFPLOAD addressing in the 32-bit dest (documented in
// tt-llk's ckernel_sfpu_binary_bcast.h and ckernel_sfpu_rope.h): one vector is 4 dest rows x 8 columns of a face,
// address bits >= 2 select the 4-row group and bit 1 the column half, so vectors 2p and 2p + 1 together are the
// face's rows 4p..4p + 3 at all 16 columns.  Faces 0 and 1 hold token rows 0-15 (experts 0-15 / 16-31 of the tile),
// faces 2 and 3 token rows 16-31, so the vector PAIR p of face f holds token rows (f >> 1) * 16 + 4p .. + 3.  A token
// row the core never produces is never compared with a live column (the sort runs inside a token column of the
// transposed tile), never summed with a live row (the reductions are row-wise) and never written, so its exp is
// unobservable.  `exp_tile_live(idst, pair_mask)` issues the same instructions as `exp_tile<false>` on the pairs
// whose bit is set (bit p for faces 0/1, bit 4 + p for faces 2/3) and only the address increment on the others: a
// live row computes bit for bit what the full call computes, at the same DST address.  The pair is the unit because
// a single vector holds only half of its four rows' experts (measured 2026-09-25 on one Blackhole chip before the
// layout was read: the LLK's own loop at ITERATIONS = 1 leaves the live row wrong and at 4 right; a per-vector guard
// at {0, 1} right, at {0} or {0, 1, 2} wrong on the odd vector).  The loop ends at the last live pair: the face step
// (`_llk_math_eltwise_sfpu_inc_dst_face_addr_`, a carriage-return SETRWC) places the next face from the CR base
// whatever the vector counter reached, exactly as `exp_tile`'s own `iterations` template argument relies on, so the
// vectors after the last live pair cost nothing (measured: the per-vector increments of 28 skipped vectors were ~3 us
// per call).  tests/test_fused_router_tail_static.py regenerates the guarded loop from the LLK source and pins it.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"

#ifdef TRISC_MATH
namespace ckernel {
namespace sfpu {

// ---- copied from ckernel_sfpu_exp.h: calculate_exponential, the precise fp32-dest branch, with the vector guard ----
template <bool is_fp32_dest_acc_en, bool SCALE_EN = false, int ITERATIONS = 8>
inline void calculate_exponential_live(
    const std::uint32_t pair_mask, const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    static_assert(
        is_fp32_dest_acc_en, "the live-vector exp is the precise fp32-dest path (the bf16 path is a replay body)");
    // vectors 0 .. 2 * (last live pair + 1) - 1; 0 when the face pair holds no live row
    int end = 0;
    for (int p = 0; p < ITERATIONS / 2; p++) {
        if ((pair_mask >> p) & 1u) {
            end = 2 * (p + 1);
        }
    }
    for (int d = 0; d < end; d++) {
        if ((pair_mask >> (d >> 1)) & 1u) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);
        }
        sfpi::dst_reg++;
    }
}
// ---- end of the copy ----

// _llk_math_eltwise_unary_sfpu_params_ + _llk_math_eltwise_sfpu_apply_vector_mode_ (VectorMode::RC) with the mask.
template <bool is_fp32_dest_acc_en>
inline void exp_tile_live_math(std::uint32_t dst_index, std::uint32_t pair_mask) {
    _llk_math_eltwise_sfpu_start_(dst_index);
    for (int face = 0; face < 4; face++) {
        calculate_exponential_live<is_fp32_dest_acc_en>((pair_mask >> (4 * (face >> 1))) & 0xFu);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

// exp_tile<false>(idst) over the live vector pairs: bit p of pair_mask = token rows 4p .. 4p + 3 (faces 0 and 1),
// bit 4 + p = token rows 16 + 4p .. 19 + 4p (faces 2 and 3).  0xFF is exp_tile<false> itself.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_live(uint32_t idst, uint32_t pair_mask) {
    MATH(
        (ckernel::_sfpu_check_<DST_SYNC_MODE>(idst, VectorMode::RC),
         ckernel::sfpu::exp_tile_live_math<is_fp32_dest_acc_en>(idst, pair_mask)));
}

// The vector pairs of a tile that carry a live token row: token r lives in pair (r % 16) / 4 of faces r / 16 * 2 and
// + 1.
ALWI uint32_t exp_live_pairs(uint32_t token_mask) {
    uint32_t pairs = 0;
    for (uint32_t p = 0; p < 4; ++p) {
        if ((token_mask >> (4 * p)) & 0xFu) {
            pairs |= 1u << p;
        }
        if ((token_mask >> (16 + 4 * p)) & 0xFu) {
            pairs |= 1u << (4 + p);
        }
    }
    return pairs;
}
