// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_add.h"
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

// True when calculate_sfpu_binary_two_row can serve the call: a single-instruction op with no
// software rounding after it, over an even row count. An odd ITERATIONS takes the one-row body
// instead; every caller passes 8 rows per face.
template <BinaryOp BINOP, bool software_rounding, int ITERATIONS>
inline constexpr bool use_two_row_binary =
    (BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL) && !software_rounding &&
    ITERATIONS % 2 == 0;

/**
 * @brief ADD/SUB/MUL over two Dest rows per iteration.
 *
 * A row that takes no software rounding is load, load, op, store, and the SFPU stalls on the
 * op -> store dependency on every row. Two rows per iteration, the pattern ckernel_sfpu_addcmul.h
 * uses, let the second row's op fill that stall: SFPLOAD x4, op x2, SFPSTORE x2 and one TTINCRWC
 * per two rows. Identical instructions on identical operands, only the issue order changes, so
 * the results are bit-identical to the one-row body.
 *
 * @tparam BINOP: ADD, SUB or MUL
 * @tparam ITERATIONS: number of sfpi rows to process (even)
 * @tparam TILE_SHAPE: destination tile shape used to calculate operand offsets
 */
template <BinaryOp BINOP, int ITERATIONS, trisc::DstTileShape TILE_SHAPE>
inline void calculate_sfpu_binary_two_row(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        use_two_row_binary<BINOP, false /*software_rounding*/, ITERATIONS>,
        "two rows per iteration: ADD/SUB/MUL over an even row count");
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat a_res;
        sfpi::vFloat b_res;
        if constexpr (BINOP == BinaryOp::ADD) {
            a_res = a_in0 + a_in1;
            b_res = b_in0 + b_in1;
        } else if constexpr (BINOP == BinaryOp::SUB) {
            a_res = a_in0 - a_in1;
            b_res = b_in0 - b_in1;
        } else {
            a_res = a_in0 * a_in1;
            b_res = b_in0 * b_in1;
        }
        sfpi::dst_reg[off_out] = a_res;
        sfpi::dst_reg[off_out + 1] = b_res;
        sfpi::dst_reg += 2;
    }
}

/**
 * @brief LLK caller for binary SFPU operations: ADD, SUB, RSUB, MUL, DIV, XLOGY, NEXTAFTER and NEXTAFTER_BF16.
 *
 * @note DIV special cases (matching BH semantics):
 *   - 0 / 0 -> NaN
 *   - x / 0 -> ±inf, sign of x
 *   - x / x -> 1.0 (forced exact, regardless of reciprocal rounding)
 * @note XLOGY is in0 * ln(in1): NaN for a negative or NaN in1, in0 * -inf for +-0 and denormal in1.
 * @note NEXTAFTER / NEXTAFTER_BF16 step in0 one fp32 / bfloat16 ULP toward in1 (in1 when they are equal); a
 *       subnormal result is flushed to zero by the store, and a NaN operand gives NaN.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: selects which binary op to compute
 * @tparam is_fp32_dest_acc_en: enables FP32 DEST accumulation (skips bf16 RNE for DIV, ADD, SUB, RSUB)
 * @tparam dst_rounding_mode: bf16 narrowing applied to ADD/SUB/RSUB results (no-op if is_fp32_dest_acc_en).
 *         DIV ignores this and always rounds RNE, to match BH semantics.
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 * @tparam TILE_SHAPE: destination tile shape used to calculate operand offsets
 */
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    BinaryOp BINOP,
    bool is_fp32_dest_acc_en,
    DstRoundingMode dst_rounding_mode = DstRoundingMode::Default,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_sfpu_binary(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::RSUB || BINOP == BinaryOp::MUL ||
            BINOP == BinaryOp::DIV || BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::NEXTAFTER ||
            BINOP == BinaryOp::NEXTAFTER_BF16,
        "calculate_sfpu_binary only supports ADD, SUB, RSUB, MUL, DIV, XLOGY, NEXTAFTER and NEXTAFTER_BF16");
    static_assert(
        dst_rounding_mode == DstRoundingMode::Default || BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB ||
            BINOP == BinaryOp::RSUB,
        "NearestEven rounding parameter is currently supported for ADD, SUB and RSUB only");
    // ADD/SUB/RSUB narrow to bf16 in software only under NearestEven on a 16-bit Dest; MUL never does
    // (no narrowing), DIV always does on a 16-bit Dest and is excluded by use_two_row_binary.
    constexpr bool round_to_bf16_rne = !is_fp32_dest_acc_en && dst_rounding_mode == DstRoundingMode::NearestEven;
    if constexpr (use_two_row_binary<BINOP, round_to_bf16_rne, ITERATIONS>) {
        calculate_sfpu_binary_two_row<BINOP, ITERATIONS, TILE_SHAPE>(dst_index_in0, dst_index_in1, dst_index_out);
        return;
    }
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    constexpr bool round_div_to_bf16 = BINOP == BinaryOp::DIV && !is_fp32_dest_acc_en;
    // The bf16 RNE constants are bound once per face, not re-materialised on every row (two
    // SFPLOADI per row otherwise). Unassigned unless a path below rounds in software, which also
    // gates their use.
    sfpi::vUInt rne_bias;
    sfpi::vUInt rne_mask;
    if constexpr (round_to_bf16_rne || round_div_to_bf16) {
        rne_bias = kBf16RneBias;
        rne_mask = kBf16RneMask;
    }
    [[maybe_unused]] constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    [[maybe_unused]] constexpr std::int32_t kAbsMask = 0x7FFFFFFF;
    [[maybe_unused]] constexpr std::int32_t kInfBits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (BINOP == BinaryOp::ADD && !round_to_bf16_rne) {
            using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
            calculate_binary_operands<AddFloatMath, 1 /*ITERATIONS*/>(
                Operand{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
                Operand{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
                Operand{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
        } else {
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
            sfpi::vFloat result = 0.0f;

            if constexpr (BINOP == BinaryOp::ADD) {
                result = in0 + in1;
            } else if constexpr (BINOP == BinaryOp::MUL) {
                result = in0 * in1;
            } else if constexpr (BINOP == BinaryOp::SUB) {
                result = in0 - in1;
            } else if constexpr (BINOP == BinaryOp::RSUB) {
                result = in1 - in0;
            } else if constexpr (BINOP == BinaryOp::DIV) {
                constexpr int reciprocal_iterations = 2;  // Two Newton-Raphson iterations
                result = in0 * _sfpu_reciprocal_<reciprocal_iterations>(in1);

                v_if(in1 == 0) {
                    v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
                    v_else {
                        result = std::numeric_limits<float>::infinity();
                        result = sfpi::copysgn(result, in0);
                    }
                    v_endif;
                }
                // sfpi's vFloat equality subtracts the operands as integers and tests the
                // difference as sign-magnitude, so it matches x == -x as well as x == x. Take the
                // magnitude from the shortcut and the sign from the quotient, correct for both.
                v_elseif(in0 == in1) { result = sfpi::copysgn(sfpi::vFloat(1.0f), result); }
                v_endif;

                if constexpr (round_div_to_bf16) {
                    // Software RNE conversion to match FPU bf16 rounding (Quasar SFPSTORE
                    // truncates by default).
                    result = float32_to_bf16_rne(result, rne_bias, rne_mask);
                }
            } else if constexpr (BINOP == BinaryOp::XLOGY) {
                // ln(in1) on the register rather than through Dest, with the log op's body. Its 32-bit Dest
                // polynomial in both Dest modes: sfpu_binary_init does not know the Dest mode, and log_init
                // programs that polynomial's constants. Zero and denormal in1 give -inf.
                sfpi::vFloat log_in1 = calculate_log_body<
                    false /*FAST_APPROX*/,
                    false /*HAS_BASE_SCALING*/,
                    true /*is_fp32_dest_acc_en*/>(in1, 0 /*log_base_scale_factor*/);
                if constexpr (!is_fp32_dest_acc_en) {
                    // Narrow the log to the 16-bit Dest's bfloat16 (SFPSTORE keeps the high half) before the
                    // product, as Blackhole does. A shift pair rather than a materialised 0xFFFF0000 mask.
                    constexpr std::uint32_t kBf16DroppedBits = 16;
                    log_in1 = sfpi::as<sfpi::vFloat>(
                        (sfpi::as<sfpi::vUInt>(log_in1) >> kBf16DroppedBits) << kBf16DroppedBits);
                }
                result = log_in1 * in0;
                // NaN for a negative in1 and for a NaN in1, as on Blackhole. The log body already returns NaN
                // there, but it flushes a negative denormal to +0 (-inf); this guard reads in1's own sign. The
                // NaN test is on the bit pattern: sfpi's vFloat equality compares integer patterns, so
                // `in1 == nan` would match one NaN encoding only. Flat v_ifs, not one `||` condition.
                v_if(in1 < 0.0f) { result = nan; }
                v_endif;
                v_if((sfpi::as<sfpi::vInt>(in1) & kAbsMask) > kInfBits) { result = nan; }
                v_endif;
            } else if constexpr (BINOP == BinaryOp::NEXTAFTER || BINOP == BinaryOp::NEXTAFTER_BF16) {
                // Step in0 one representable value toward in1 on its bit pattern: consecutive floats of one
                // sign are consecutive integers, so that is one ULP at in0's own magnitude. bfloat16 keeps its
                // mantissa in the top 16 bits of the fp32 register, so one of its ULPs is 0x10000 there.
                constexpr int kUlpStep = (BINOP == BinaryOp::NEXTAFTER_BF16) ? 0x10000 : 1;
                // Flat v_ifs, every value declared up front: the predication pass does not survive nesting here.
                sfpi::vInt bits = sfpi::as<sfpi::vInt>(in0);
                sfpi::vInt step = 0;  // in0 == in1 keeps in0
                // The pattern grows away from zero for either sign, so the direction depends on in0's sign too.
                v_if(in0 < in1 && in0 >= 0.0f) { step = kUlpStep; }
                v_endif;
                v_if(in0 < in1 && in0 < 0.0f) { step = -kUlpStep; }
                v_endif;
                v_if(in0 > in1 && in0 > 0.0f) { step = -kUlpStep; }
                v_endif;
                v_if(in0 > in1 && in0 <= 0.0f) { step = kUlpStep; }
                v_endif;
                result = sfpi::as<sfpi::vFloat>(bits + step);
                // Zeros: neither zero reaches its neighbour by stepping its own pattern, and the answer's sign
                // comes from in1. Tested on the magnitudes so that -0.0 takes the same path as +0.0.
                sfpi::vFloat mag_a = sfpi::setsgn(in0, 0);
                sfpi::vFloat mag_b = sfpi::setsgn(in1, 0);
                sfpi::vFloat tiny = sfpi::as<sfpi::vFloat>(sfpi::vInt(kUlpStep));
                v_if(mag_a == 0.0f && in1 > 0.0f) { result = tiny; }
                v_endif;
                v_if(mag_a == 0.0f && in1 < 0.0f) { result = -tiny; }
                v_endif;
                // Equal zeros return in1's zero: nextafter(+0, -0) is -0.
                v_if(mag_a == 0.0f && mag_b == 0.0f) { result = in1; }
                v_endif;
                // A NaN operand propagates; classified on the integer pattern (exponent all ones, non-zero
                // mantissa), last so it wins over the direction and zero arms.
                v_if((bits & kAbsMask) > kInfBits) { result = nan; }
                v_endif;
                v_if((sfpi::as<sfpi::vInt>(in1) & kAbsMask) > kInfBits) { result = nan; }
                v_endif;
            }

            if constexpr (
                (BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::RSUB) && round_to_bf16_rne) {
                result = float32_to_bf16_rne(result, rne_bias, rne_mask);
            }

            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        }
        sfpi::dst_reg++;
    }
}

/**
 * @brief calculate_sfpu_binary in the Blackhole template order, <APPROX, BINOP, ITERATIONS, is_fp32_dest_acc_en>.
 *
 * The Compute API headers shared with Blackhole (xlogy.h, nextafter.h) spell the call in that order; this forwards it
 * to the Quasar order above. Each call resolves to exactly one of the two: an int ITERATIONS cannot bind to the
 * Quasar template's bool is_fp32_dest_acc_en, nor a bool to its DstRoundingMode, and a call that names only three
 * arguments leaves this one's is_fp32_dest_acc_en undeduced.
 *
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 */
template <
    bool APPROXIMATION_MODE,
    BinaryOp BINOP,
    int ITERATIONS,
    bool is_fp32_dest_acc_en,
    DstRoundingMode dst_rounding_mode = DstRoundingMode::Default,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_sfpu_binary(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_sfpu_binary<APPROXIMATION_MODE, BINOP, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS, TILE_SHAPE>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

/**
 * @brief Float binary compare on Dest: y = (x0 OP x1) ? 1.0f : 0.0f, OP one of SfpuType::eq/ne/lt/gt/le/ge.
 *
 * IEEE semantics: -0 and +0 compare equal, a NaN operand makes every ordering and eq false and ne true.
 * Width-agnostic (default sfpmem mode), so it serves a 16-bit and a 32-bit Dest alike.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH signature
 * @tparam ITERATIONS: number of sfpi rows to process (one call per face)
 * @tparam OP: the comparison
 * @tparam TILE_SHAPE: destination tile shape used to calculate operand offsets
 */
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType OP,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_binary_comp_fp32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        OP == SfpuType::eq || OP == SfpuType::ne || OP == SfpuType::lt || OP == SfpuType::gt || OP == SfpuType::le ||
            OP == SfpuType::ge,
        "calculate_binary_comp_fp32 supports eq, ne, lt, gt, le and ge");
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    // gt/ge are lt/le with the operands swapped.
    constexpr bool swap_operands = OP == SfpuType::gt || OP == SfpuType::ge;
    const std::uint32_t idx_a = (swap_operands ? dst_index_in1 : dst_index_in0) * dst_tile_size_sfpi;
    const std::uint32_t idx_b = (swap_operands ? dst_index_in0 : dst_index_in1) * dst_tile_size_sfpi;
    const std::uint32_t idx_out = dst_index_out * dst_tile_size_sfpi;
    constexpr std::int32_t kAbsMask = 0x7FFFFFFF;
    constexpr std::int32_t kInfBits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat a = sfpi::dst_reg[idx_a];
        const sfpi::vFloat b = sfpi::dst_reg[idx_b];
        const sfpi::vInt mag_a = sfpi::as<sfpi::vInt>(a) & kAbsMask;
        const sfpi::vInt mag_b = sfpi::as<sfpi::vInt>(b) & kAbsMask;
        // Flat v_ifs: the sfpi predication pass does not survive || nested around && in one condition.
        sfpi::vFloat result = 0.0f;
        if constexpr (OP == SfpuType::eq || OP == SfpuType::ne) {
            v_if(a == b) { result = 1.0f; }
            v_endif;
        } else if constexpr (OP == SfpuType::lt || OP == SfpuType::gt) {
            v_if(a < b) { result = 1.0f; }
            v_endif;
        } else {
            v_if(a <= b) { result = 1.0f; }
            v_endif;
        }
        // -0 and +0 are equal: not ordered strictly, equal for eq / le / ge.
        constexpr float both_zero_result = (OP == SfpuType::lt || OP == SfpuType::gt) ? 0.0f : 1.0f;
        v_if(mag_a == 0 && mag_b == 0) { result = both_zero_result; }
        v_endif;
        // A NaN operand is neither ordered nor equal; ne inverts after the guard.
        v_if(mag_a > kInfBits) { result = 0.0f; }
        v_endif;
        v_if(mag_b > kInfBits) { result = 0.0f; }
        v_endif;
        if constexpr (OP == SfpuType::ne) {
            result = 1.0f - result;
        }
        sfpi::dst_reg[idx_out] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Initialisation hook for binary SFPU kernels.
 * For DIV, programs the Newton-Raphson reciprocal constant; for XLOGY, the 32-bit Dest log constants in
 * vConstFloatPrgm0-2 (log_init); no-op for the other ops.
 *
 * @tparam APPROXIMATION_MODE: forwarded to the op-specific init
 * @tparam BINOP: selects which op's init to run
 */
template <[[maybe_unused]] bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::DIV) {
        // DIV always runs the Newton-refined _sfpu_reciprocal_<2>, so the Newton constant is
        // needed in every mode, not just the non-approximate one.
        _init_reciprocal_</*APPROXIMATION_MODE=*/false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        log_init<APPROXIMATION_MODE, false /*FAST_APPROX*/, true /*is_fp32_dest_acc_en*/>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
