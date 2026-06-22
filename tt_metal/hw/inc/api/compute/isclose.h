// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_isclose.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise isclose operation: result = |a - b| <= atol + rtol * |b|
 *
 * The tolerance scalars ``rtol`` and ``atol`` are passed as their IEEE-754
 * bit-patterns (``uint32_t``) and converted to ``float`` inside the kernel via
 * ``Converter::as_float()``. Callers typically obtain these via
 * ``std::bit_cast<uint32_t>(rtol_float)`` or ``get_arg_val<uint32_t>(idx)``.
 *
 * The ``EQUAL_NAN`` template parameter controls NaN semantics (mirrors
 * ``torch.isclose(equal_nan=...)``):
 *   - ``false`` (default, matches torch): any NaN input => result = 0
 *   - ``true``:  both NaN => result = 1; one NaN => result = 0
 *
 * Inputs must be float32 or bfloat16. INT32 tensors must be promoted to
 * FLOAT32 before reaching this kernel; invoke_binary_ng_isclose handles
 * this via explicit ttnn::typecast calls before dispatch.
 *
 * INT32 precision: FLOAT32 has a 24-bit significand, so converting INT32
 * to FLOAT32 is lossy when |x| exceeds about 2^24 (~1.67e7). Comparison runs
 * on the rounded operands and can disagree with ``torch.isclose`` on the
 * original integers (e.g. two distinct INT32 values that round to the same
 * float may compare as close even when rtol = atol = 0).
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
 * | EQUAL_NAN      | NaN comparison semantics (see above)              | true / false      | False    |
 *
 * | Argument  | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0     | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1     | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst      | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | rtol_bits | IEEE-754 bit-pattern of the relative tolerance scalar                 | uint32_t | Must represent a finite non-negative float            | True     |
 * | atol_bits | IEEE-754 bit-pattern of the absolute tolerance scalar                 | uint32_t | Must represent a finite non-negative float            | True     |
 */
// clang-format on
template <bool EQUAL_NAN = false>
ALWI void isclose_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, uint32_t rtol_bits, uint32_t atol_bits) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_isclose,
        (APPROX, 8 /* ITERATIONS */, EQUAL_NAN),
        idst0,
        idst1,
        odst,
        VectorMode::RC,
        rtol_bits,
        atol_bits)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void isclose_binary_tile_init() { MATH((SFPU_BINARY_INIT(isclose))); }

}  // namespace ckernel
