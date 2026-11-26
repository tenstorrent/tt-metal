// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-row softmax over the first k columns (1 <= k <= FACE_C_DIM) of an (M, 16) or (M, 32)
// tile (M in {1, 2, 4, 8}) residing in DST via standard copy_tile.
//
// Handles unsorted or presorted inputs, zero logits inside [0, k), arbitrary odd or even k, and
// dirty tail elements in [k, FACE_C_DIM) (masked before max and zeroed after exp). Each row is
// normalized independently:
//   M in {1, 2, 4}: 1 SFPU row-group (rows 0..3)
//   M == 8:         2 SFPU row-groups (rows 0..3, rows 4..7)
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_row_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel::sfpu::softmax_k_detail {

using row_common::store;
using row_common::subvec_max;
using row_common::subvec_sum;
using row_common::valid_num_rows;

template <std::uint32_t NUM_ROWS>
constexpr int kRowGroups = (NUM_ROWS == 8) ? 2 : 1;

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat eval_exp(sfpi::vFloat x) {
    if constexpr (is_fp32_dest_acc_en) {
        return _sfpu_exp_fp32_accurate_(x);
    } else {
        return _sfpu_exp_21f_bf16_<false>(x);
    }
}

}  // namespace ckernel::sfpu::softmax_k_detail

namespace ckernel::sfpu {

// [[gnu::noinline]] keeps GCC's rvtt_replay buffer scoped to this leaf SFPU call. Ends with
// STALL_MATH on WAIT_SFPU because _llk_math_eltwise_sfpu_done_() only clears dst_reg_addr and
// follow-on FPU ops do not wait on SFPU.
template <bool is_fp32_dest_acc_en, std::uint32_t NUM_ROWS>
[[gnu::noinline]] inline void _calculate_softmax_k_(const std::uint32_t k) {
    static_assert(softmax_k_detail::valid_num_rows<NUM_ROWS>, "softmax_k: NUM_ROWS must be 1, 2, 4, or 8");
    LLK_ASSERT(k >= 1u && k <= FACE_C_DIM, "softmax_k: k must be in [1, FACE_C_DIM]");

    // Largest negative finite BF16 value (-3.38953139e38f): finite in both BF16 and FP32 so
    // subtracting row_max never creates NaN for finite active logits.
    constexpr float kNegLarge = -3.38953139e38f;
    constexpr int kEvenColMask = static_cast<int>(FACE_C_DIM - 2);

#pragma GCC unroll 2
    for (int g = 0; g < softmax_k_detail::kRowGroups<NUM_ROWS>; ++g) {
        const int v = 2 * g;
        sfpi::vFloat x_even = sfpi::dst_reg[v + 0];
        sfpi::vFloat x_odd = sfpi::dst_reg[v + 1];

        if (k < FACE_C_DIM) {
            // Blackhole vConstTileId = 2 * lane; (vConstTileId & (FACE_C_DIM - 2)) gives even
            // column 0,2,...,14 in every 4-row subvector. Odd vector holds even_col + 1.
            sfpi::vInt even_col = sfpi::vConstTileId & kEvenColMask;
            v_if(even_col >= static_cast<std::int32_t>(k)) { x_even = kNegLarge; }
            v_endif;
            v_if(even_col >= static_cast<std::int32_t>(k - 1)) { x_odd = kNegLarge; }
            v_endif;
        }

        sfpi::vFloat row_max = softmax_k_detail::subvec_max(sfpi::max(x_even, x_odd));

        sfpi::vFloat e_even = softmax_k_detail::eval_exp<is_fp32_dest_acc_en>(x_even - row_max);
        sfpi::vFloat e_odd = softmax_k_detail::eval_exp<is_fp32_dest_acc_en>(x_odd - row_max);

        if (k < FACE_C_DIM) {
            sfpi::vInt even_col = sfpi::vConstTileId & kEvenColMask;
            v_if(even_col >= static_cast<std::int32_t>(k)) { e_even = 0.0f; }
            v_endif;
            v_if(even_col >= static_cast<std::int32_t>(k - 1)) { e_odd = 0.0f; }
            v_endif;
        }

        // row_sum >= 1.0f because at least one active element equals row_max (exp(0) == 1).
        sfpi::vFloat row_sum = softmax_k_detail::subvec_sum(e_even + e_odd);
        sfpi::vFloat inv_sum = sfpi::approx_recip(row_sum);
        inv_sum = inv_sum * (2.0f - row_sum * inv_sum);
        if constexpr (is_fp32_dest_acc_en) {
            inv_sum = inv_sum * (2.0f - row_sum * inv_sum);
        }

        softmax_k_detail::store<is_fp32_dest_acc_en>(v + 0, e_even * inv_sum);
        softmax_k_detail::store<is_fp32_dest_acc_en>(v + 1, e_odd * inv_sum);
    }

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

}  // namespace ckernel::sfpu
