// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
// Reciprocal sfpi helper (_sfpu_reciprocal_ / _init_sfpu_reciprocal_) still lives in the
// tt-llk common layer; only the binary *wrapper* has been inlined into this L4 header.
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Converts float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
 * Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking, ported
 * from BH. Applied before SFPSTORE because Quasar truncates by default (BH FPU
 * exposes RNE for bf16 store natively).
 *
 * @param in: float32 value to convert
 * @return bf16 value packed in the upper 16 bits of a float32
 */
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32).
    // Needed for the tie-breaker: round to even.
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - lower 16 bits > 0x8000      -> overflow, rounds up
    // - lower 16 bits < 0x8000      -> no overflow, rounds down
    // - lower 16 bits == 0x8000 (tie)
    //     and lsb=0: 0x7fff+0=0xffff, no overflow -> stays even
    //     and lsb=1: 0x7fff+1=0x8000,    overflow -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32).
    bits = bits & 0xFFFF0000U;

    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

/**
 * @brief Vectorized binary multiply (in0 * in1), float16b inputs/output.
 *
 * Mirrors the Blackhole metal implementation. When the dest is not in fp32
 * accumulation mode the float32 product is converted to bf16 with software RNE
 * (Quasar SFPSTORE truncates by default), and the FPU 0*x = 0 / x*0 = 0
 * semantics are forced explicitly.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: unused for MUL, preserved for API parity
 * @tparam ITERATIONS: number of sfpi row-pairs to process per face
 * @tparam is_fp32_dest_acc_en: enables FP32 DEST accumulation (skips bf16 RNE)
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // Software RNE conversion to match FPU bf16 rounding (Quasar SFPSTORE truncates).
            // NB: unlike BH we do not add an explicit 0*x=0 guard — IEEE multiply already
            // yields 0 for 0*finite, and the extra sfpi v_if regressed results on Quasar.
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Vectorized binary divide (in0 / in1), float16b inputs/output.
 * Ported from BH (`ckernel_sfpu_binary.h`, `calculate_sfpu_binary_div`).
 *
 * @note Special cases (matching BH semantics):
 *   - 0 / 0 -> NaN
 *   - x / 0 -> ±inf, sign of x
 *   - x / x -> 1.0 (forced exact, regardless of reciprocal rounding)
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: unused for DIV, preserved for API parity
 * @tparam ITERATIONS: number of sfpi row-pairs to process per face
 * @tparam is_fp32_dest_acc_en: enables FP32 DEST accumulation (skips bf16 RNE)
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        constexpr int reciprocal_iterations = 2;  // Two Newton-Raphson iterations
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<reciprocal_iterations>(in1);

        v_if(in1 == 0) {
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::copysgn(result, in0);
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // Software RNE conversion to match FPU bf16 rounding (Quasar SFPSTORE truncates).
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Initialisation hook for binary SFPU kernels.
 * For DIV, programs the Newton-Raphson reciprocal constant; no-op for MUL.
 *
 * @tparam APPROXIMATION_MODE: forwarded to the op-specific init
 * @tparam BINOP: selects which op's init to run
 */
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::DIV) {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
