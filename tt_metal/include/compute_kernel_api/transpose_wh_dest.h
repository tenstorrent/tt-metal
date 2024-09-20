// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_transpose_dest_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif


namespace ckernel {

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose_wh_dest to be executed correctly.
 */
ALWI void transpose_wh_dest_init_short()
{
    MATH(( llk_math_transpose_dest_init() ));
}

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
ALWI void transpose_wh_dest(uint32_t idst)
{
    UNPACK(( llk_unpack_set_srcb_dummy_valid() ));
    MATH(( llk_math_transpose_dest(idst) ));
}

} // namespace ckernel
