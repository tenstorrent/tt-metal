// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_transpose_dest_api.h"
#else
#include "llk_math_transpose_dest.h"
#endif
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose_wh_dest to be
 * executed correctly.
 */
template <bool is_32bit = false>
ALWI void transpose_wh_dest_init_short() {
#ifndef ARCH_QUASAR
    MATH((llk_math_transpose_dest_init<true, is_32bit>()));
#else
    // Derive EN_32BIT_DEST at compile time: explicit is_32bit wins, otherwise
    // fall back to DST_ACCUM_MODE so 32-bit-DEST kernels (Float32 / Int32 via
    // unpack-to-dest) automatically get the 32-bit MOVD2B MOP strides without
    // needing every kernel to remember to pass is_32bit=true.
    constexpr bool EN_32BIT_DEST = is_32bit || DST_ACCUM_MODE;
    MATH((_llk_math_transpose_dest_init_<true /*TRANSPOSE_OF_FACES*/, EN_32BIT_DEST>()));
#endif
}

// clang-format off
/**
 * Performs a 32x32 in place transpose operation *B[w,h] = A[h,w]* on a tile in the DST register at idst.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | idst           | The index of the tile in DST REG to transpose           | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
 // clang-format on
template <bool is_32bit = false>
ALWI void transpose_wh_dest(uint32_t idst) {
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_transpose_dest<true, is_32bit>(idst)));
#else
    // Quasar: data is already in DEST from the preceding copy_tile; the transpose
    // is an in-place MATH operation on DEST, so no UNPACK work is needed here.
    // Note: llk_unpack_set_srcb_dummy_valid() (used on the non-Quasar path) is
    // not required on Quasar because the Layer-1 transpose helper already issues
    // STALLWAIT SRCB_VLD internally to gate the SrcB-consuming instructions.
    MATH((_llk_math_transpose_dest_(idst)));
#endif
}

}  // namespace ckernel
