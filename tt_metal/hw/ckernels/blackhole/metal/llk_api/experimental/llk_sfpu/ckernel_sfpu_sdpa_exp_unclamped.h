// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
// The leaf header reaches for the exp helpers unqualified ("ckernel_sfpu_exp.h"); they live
// in llk_api/llk_sfpu, one layer up from tt-llk. Pulled in here under their on-path spelling
// so the dependency is visible at the point that needs it.
#include "ckernel_sfpu_exp.h"
#include "sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h"

namespace ckernel::sfpu {

/**
 * @brief Op init for calculate_sdpa_exp_unclamped: ADDR_MOD_6 (dest += 2) and LREG12/13 (1/ln2, c2).
 *
 * Run once after the invariant SFPU init; re-run after any op init that reprograms LREG12/13 or
 * ADDR_MOD_6 (exp_init<true>, sfpu_reciprocal_init, log_init, ...). Same register contract as
 * exp_init<false> for the shared clamped TTI exp, so the two may share a thread's state.
 */
inline void sdpa_exp_unclamped_init() { _init_sdpa_exp_unclamped_(); }

/**
 * @brief Exponentiate one DEST face in place, without the upper input clamp.
 *
 * Drives @ref _calculate_sdpa_exp_unclamped_ -- the replayed TTI twin of the shared clamped
 * exp_21f -- over the 8 SFPU slots of a 16x16 face, in the shape @ref
 * _llk_math_eltwise_unary_sfpu_params_ / SFPU_UNARY_CALL expects; VectorMode::RC repeats it
 * over the four faces.
 *
 * @tparam SCALE_EN: Multiply the input by exp_base_scale_factor first, values = <true/false>
 * @param exp_base_scale_factor: Scale as a raw bf16 bit pattern; ignored when SCALE_EN is false.
 * @note bf16 DEST only -- the kernel rounds fp32->bf16 before every store unconditionally.
 * @note Callers must pass val <= 0, which is what makes dropping the upper clamp safe. The
 *       clamped path saturates xlog2 = val/ln2 + 127 at its upper bound; that bound is
 *       unreachable for non-positive inputs, so removing it is dead-code removal for the SDPA
 *       use case and a wrap in the float->int step for anything above val*scale ~= 88.7.
 * @note Requires @ref sdpa_exp_unclamped_init and clobbers replay slot 0.
 */
template <bool SCALE_EN, bool is_fp32_dest_acc_en>
inline void calculate_sdpa_exp_unclamped(const std::uint32_t exp_base_scale_factor) {
    static_assert(!is_fp32_dest_acc_en, "upper-unclamped exp variant implemented for bf16 dest only");
    // One SFPU slot is 4 DEST rows x 8 columns, so a full 16x16 face is 8 slots.
    constexpr int ITERATIONS_FULL_FACE = 8;
    _calculate_sdpa_exp_unclamped_<SCALE_EN, ITERATIONS_FULL_FACE>(static_cast<std::uint16_t>(exp_base_scale_factor));
}

}  // namespace ckernel::sfpu
