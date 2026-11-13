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
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief LLK caller for binary SFPU operations, currently supports ADD, SUB, RSUB, MUL and DIV.
 *
 * @note DIV special cases (matching BH semantics):
 *   - 0 / 0 -> NaN
 *   - x / 0 -> ±inf, sign of x
 *   - x / x -> 1.0 (forced exact, regardless of reciprocal rounding)
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: selects which binary op to compute (ADD, SUB, RSUB, MUL or DIV)
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
            BINOP == BinaryOp::DIV,
        "calculate_sfpu_binary only supports ADD, SUB, RSUB, MUL and DIV");
    static_assert(
        dst_rounding_mode == DstRoundingMode::Default || BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB ||
            BINOP == BinaryOp::RSUB,
        "NearestEven rounding parameter is currently supported for ADD, SUB and RSUB only");
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (BINOP == BinaryOp::ADD) {
            using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
            constexpr bool round_to_bf16 = !is_fp32_dest_acc_en && dst_rounding_mode == DstRoundingMode::NearestEven;
            using Output = SfpuOperand<SfpuReg::Dest, DestBf16RneFormat<round_to_bf16>>;
            calculate_binary_operands<AddFloatMath, 1 /*ITERATIONS*/>(
                Operand{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
                Operand{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
                Output{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
        } else {
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
            sfpi::vFloat result = 0.0f;

            if constexpr (BINOP == BinaryOp::MUL) {
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

                if constexpr (!is_fp32_dest_acc_en) {
                    // Software RNE conversion to match FPU bf16 rounding (Quasar SFPSTORE
                    // truncates by default).
                    result = float32_to_bf16_rne(result);
                }
            }

            if constexpr (
                (BINOP == BinaryOp::SUB || BINOP == BinaryOp::RSUB) && !is_fp32_dest_acc_en &&
                dst_rounding_mode == DstRoundingMode::NearestEven) {
                result = float32_to_bf16_rne(result);
            }

            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        }
        sfpi::dst_reg++;
    }
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
 * For DIV, programs the Newton-Raphson reciprocal constant; no-op for MUL.
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
    }
}

}  // namespace sfpu
}  // namespace ckernel
