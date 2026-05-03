// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_isclose.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise isclose operation: result = |a - b| <= atol + rtol * |b|
 *
 * The tolerance scalars ``rtol`` and ``atol`` are injected at kernel compile time
 * via the ``ISCLOSE_RTOL_VAL`` and ``ISCLOSE_ATOL_VAL`` preprocessor defines.
 *
 * The ``EQUAL_NAN`` template parameter controls NaN semantics (mirrors
 * ``torch.isclose(equal_nan=...)``):
 *   - ``false`` (default): any NaN input ⇒ result = 0
 *   - ``true``:            both NaN      ⇒ result = 1; one NaN ⇒ result = 0
 *
 * Inputs must be float32 or bfloat16. INT32 tensors must be promoted to
 * FLOAT32 before reaching this kernel; invoke_binary_ng_isclose handles
 * this via explicit ttnn::typecast calls before dispatch.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once
 * when using 16-bit formats; this is reduced to 2 tiles for 32-bit formats.
 *
 * Return value: None
 *
 * | Template Param | Description                                       | Valid Values      | Required |
 * |----------------|---------------------------------------------------|-------------------|----------|
 * | EQUAL_NAN      | NaN comparison semantics (see above)              | true / false      | True     |
 *
 * | Argument | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0    | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1    | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst     | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool EQUAL_NAN>
ALWI void isclose_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_isclose<APPROX, EQUAL_NAN>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isclose_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_isclose_init<APPROX>())); }

}  // namespace ckernel
