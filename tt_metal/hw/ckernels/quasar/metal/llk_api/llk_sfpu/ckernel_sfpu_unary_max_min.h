// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// INT32_MIN as a bit pattern. Sign-magnitude has no encoding for it (0x80000000 is -0 there).
constexpr std::uint32_t UNARY_MAX_MIN_INT32_MIN_BITS = 0x80000000u;

/**
 * @brief Element-wise max/min of a Dest tile against one uniform scalar: out = max(x, value) or min(x, value).
 *
 * The result is always one of the two operands verbatim (no arithmetic, no rounding).
 *
 * Both paths are a single SFPSWAP per row: sfpi lowers min/max to SFPSWAP with the compare domain taken
 * from the vector type (imm12 bit 0, tt_llk_quasar/instructions/assembly.yaml), so
 *   - vFloat compares as fp32 (imm12 = 1), which orders both-negative pairs correctly and follows the
 *     SFPU total order (-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN);
 *   - vInt compares as two's-complement int32 (imm12 = 0), which is exact for every int32 pair.
 * This is sfpi >= 7.83 behaviour (sfpi_lib.h min_max passes SFPSWAP_IMM_TYPE_FLOAT for vFloat); the
 * objdump of both instantiations shows `sfpswap ...,1,1` (float) and `sfpswap ...,0,1` (Int32). The
 * note in ckernel_sfpu_gelu.h that sfpi omits the float-compare bit predates it. On an older sfpi, the
 * float path would mis-order both-negative pairs; the negative-scalar test variants catch that.
 *
 * @tparam IS_MAX_OP: true selects max, false selects min.
 * @tparam FMT: math-side DataFormat. Int32 takes the integer path; every float format takes the fp32
 *         path, so callers may pass Float32 for any float Dest width (the DEFAULT load resolves it).
 * @tparam APPROXIMATION_MODE: unused (the select is exact); kept so the dispatcher's
 *         (..., APPROX, ITERATIONS) template tail matches the other Quasar SFPU kernels.
 * @tparam ITERATIONS: number of SFP row-pairs per face.
 * @tparam SIGN_MAGNITUDE_FORMAT: Int32 only. false (default): Dest holds two's-complement int32, as
 *         unpack-to-dest leaves an Int32 L1 tile. true: Dest holds sign-magnitude int32 (e.g. the FPU
 *         path); the load/store convert SM <-> two's complement around the compare.
 * @param value: scalar to compare against: an fp32 bit pattern for float FMT, a two's-complement int32
 *        for DataFormat::Int32 (in both Dest encodings). With SIGN_MAGNITUDE_FORMAT it must not be
 *        INT32_MIN, which has no sign-magnitude encoding.
 * @note No init call is required.
 * @note No SFPNOP after the SFPSWAP: the Quasar scoreboard stalls the SFPSTORE (or SFPCAST) that reads its
 *       result. TEN-4581 / TEN-4605 list the dependents it misses after a 2-cycle op (SFPNONLINEAR
 *       mode 3-5, SFPIADD/SFPSHFT, SFPCONFIG, SFPSHFT2 mode 2-4, SFPSWAP), and sfpi pads exactly those.
 */
template <
    bool IS_MAX_OP,
    DataFormat FMT,
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_unary_max_min(const std::uint32_t value) {
    static_assert(
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
            FMT == DataFormat::Tf32 || FMT == DataFormat::MxFp8R || FMT == DataFormat::MxFp8P ||
            FMT == DataFormat::Int32,
        "Unsupported DataFormat for calculate_unary_max_min().");
    static_assert(!SIGN_MAGNITUDE_FORMAT || FMT == DataFormat::Int32, "SIGN_MAGNITUDE_FORMAT applies to Int32 only.");

    if constexpr (FMT == DataFormat::Int32) {
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            LLK_ASSERT(
                value != UNARY_MAX_MIN_INT32_MIN_BITS,
                "calculate_unary_max_min: INT32_MIN has no sign-magnitude encoding");
        }
        // SM32 makes sfpi wrap the load/store in SFPCAST SM <-> two's complement; I32 is a raw copy.
        constexpr sfpi::DataLayout layout = SIGN_MAGNITUDE_FORMAT ? sfpi::DataLayout::SM32 : sfpi::DataLayout::I32;
        const sfpi::vInt s = static_cast<std::int32_t>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt x = sfpi::dst_reg[0].mode<layout>();
            sfpi::dst_reg[0].mode<layout>() = IS_MAX_OP ? sfpi::max(x, s) : sfpi::min(x, s);
            sfpi::dst_reg++;
        }
    } else {
        const sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::vUInt(value));
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = IS_MAX_OP ? sfpi::max(x, s) : sfpi::min(x, s);
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
