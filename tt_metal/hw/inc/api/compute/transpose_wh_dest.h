// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_transpose_dest_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose_wh_dest to be
 * executed correctly. On Quasar the operand's data format programs the transpose ALU config for 32-bit DST; on other
 * architectures the operand is unused.
 *
 * Return value: None
 *
 * | Argument | Description                                                | Type     | Valid Range | Required |
 * |----------|------------------------------------------------------------|----------|-------------|----------|
 * | operand  | The identifier of the circular buffer (CB) holding the data| uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool is_32bit = false, bool transpose_of_faces = true>
ALWI void transpose_wh_dest_init_short([[maybe_unused]] uint32_t operand) {
#ifndef ARCH_QUASAR
    MATH((llk_math_transpose_dest_init<transpose_of_faces, is_32bit>()));
#else
    MATH((llk_math_transpose_dest_init<transpose_of_faces, is_32bit>(operand)));
#endif
}

// clang-format off
/**
 * Performs a 32x32 in place transpose operation *B[w,h] = A[h,w]* on a tile in the DST register at idst.
 * Set transpose_of_faces=false to run only the inner face transpose used by 32-bit DST materialization paths.
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
template <bool is_32bit = false, bool transpose_of_faces = true>
ALWI void transpose_wh_dest(uint32_t idst) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
#ifndef ARCH_QUASAR
    MATH((llk_math_transpose_dest<transpose_of_faces, is_32bit>(idst)));
#else
    MATH((llk_math_transpose_dest(idst)));
#endif
}

}  // namespace ckernel
