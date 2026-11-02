// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifndef ARCH_QUASAR
#include "sfpu_mode.h"
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// Strongly typed so removed legacy_compat Boolean arguments cannot silently select DEST precision.
enum class ReciprocalDestAcc { BF16, FP32 };
enum class ReciprocalApproxMode { Default, Precise, Approximate };

/**
 * Please refer to documentation for any_init.
 * DEST precision defaults to DST_ACCUM_MODE. Explicit overrides must use ReciprocalDestAcc,
 * e.g. recip_tile_init<ReciprocalDestAcc::FP32>(); use the same mode for recip_tile.
 * approximation defaults to the kernel's APPROX setting; override it for callers that need a more
 * accurate reciprocal without changing the approximation of other operations in the kernel.
 * On Wormhole, precise BF16 reciprocal is correctly rounded only with the seed loaded here;
 * other SFPU inits may overwrite it, so call recip_tile_init again before recip_tile after them.
 * The former legacy_compat Boolean template argument is no longer accepted.
 */
template <
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16,
    ReciprocalApproxMode approximation = ReciprocalApproxMode::Default>
ALWI void recip_tile_init() {
    [[maybe_unused]] constexpr bool is_fp32_dest_acc_en = dest_acc == ReciprocalDestAcc::FP32;
    MATH(constexpr bool approximate = approximation == ReciprocalApproxMode::Default
                                          ? APPROX
                                          : approximation == ReciprocalApproxMode::Approximate;)
#ifdef ARCH_WORMHOLE
    // Match the explicit BF16 rounding in calculate_reciprocal. Other users
    // of the shared reciprocal helper retain its default minimax constants.
    MATH(SFPU_UNARY_INIT_FN(
        reciprocal, sfpu::recip_init, (approximate, is_fp32_dest_acc_en, !approximate && !is_fp32_dest_acc_en)));
#else
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (approximate, is_fp32_dest_acc_en)));
#endif
}
// clang-format off
/**
 * Performs element-wise computation of the reciprocal on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 * Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
 * DEST precision defaults to DST_ACCUM_MODE and approximation defaults to APPROX.
 * Explicit destination and approximation overrides must match recip_tile_init.
 * Legacy Boolean template arguments are no longer accepted.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (e.g., Row, Column). (default: VectorMode::RC) | VectorMode | Subject to specific hardware/kernel limits          | False    |
 */
// clang-format on
template <
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16,
    ReciprocalApproxMode approximation = ReciprocalApproxMode::Default>
ALWI void recip_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    [[maybe_unused]] constexpr bool is_fp32_dest_acc_en = dest_acc == ReciprocalDestAcc::FP32;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_reciprocal,
        (approximation == ReciprocalApproxMode::Default ? APPROX : approximation == ReciprocalApproxMode::Approximate,
         is_fp32_dest_acc_en,
         8 /*ITERATIONS*/),
        idst,
        vector_mode));
}

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise 1/x with the instruction sequence selected by *MODE*. Must follow *recip_tile_init* with the
 * same *MODE* and *dest_acc*. The DST register buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine.
 *
 * | MODE                          | Blackhole                                           | Wormhole                              |
 * |-------------------------------|-----------------------------------------------------|---------------------------------------|
 * | SfpuMode::PreciseRelaxed      | SFPARECIP + Newton pipeline (bf16: 3c, fp32: 5c/row) | 2 Newton steps (bf16: 1 + rounding)   |
 * | SfpuMode::ApproximateRelaxed  | SFPARECIP alone, 7-bit, 1 cycle per row              | 1 Newton step                         |
 * | SfpuMode::PreciseStrict       | SFPARECIP + 2 Newton steps in sfpi, NaN-guarded      | as PreciseRelaxed, NaN-guarded        |
 * | SfpuMode::ApproximateStrict   | SFPARECIP alone in sfpi, NaN-guarded                 | as ApproximateRelaxed, NaN-guarded    |
 *
 * All modes: +-0 -> +-inf, +-inf -> +-0, |x| >= 2^126 -> +-0. Strict adds NaN -> NaN; Relaxed returns 0 for NaN,
 * so its inputs must not contain NaN. The legacy *recip_tile* runs the Relaxed sequence its approximation selects.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type              | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|-------------------|-------------------------------------------------------|----------|
 * | MODE            | Instruction sequence to run (template argument)                            | SfpuMode          | Any SfpuMode value                                    | True     |
 * | dest_acc        | Dest register width (template argument)                                    | ReciprocalDestAcc | Must match the kernel's DST_ACCUM_MODE                | False    |
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t          | Must be less than the size of the DST register buffer | True     |
 * | vector_mode     | Which faces of the tile to process                                         | VectorMode        | VectorMode::RC (default), R or C                      | False    |
 */
// clang-format on
template <
    SfpuMode MODE,
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16>
ALWI void recip_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    [[maybe_unused]] constexpr bool is_fp32_dest_acc_en = dest_acc == ReciprocalDestAcc::FP32;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_reciprocal, (MODE, is_fp32_dest_acc_en), idst, vector_mode));
}

/**
 * Please refer to documentation for any_init. *MODE* and *dest_acc* must match the paired *recip_tile* call.
 */
template <
    SfpuMode MODE,
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16>
ALWI void recip_tile_init() {
    [[maybe_unused]] constexpr bool is_fp32_dest_acc_en = dest_acc == ReciprocalDestAcc::FP32;
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (MODE, is_fp32_dest_acc_en)));
}
#endif
}  // namespace ckernel
