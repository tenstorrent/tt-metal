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
 * @brief LLK caller for binary SFPU operations, currently supports ADD, SUB, MUL and DIV.
 *
 * @note DIV special cases (matching BH semantics):
 *   - 0 / 0 -> NaN
 *   - x / 0 -> ±inf, sign of x
 *   - x / x -> 1.0 (forced exact, regardless of reciprocal rounding)
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam BINOP: selects which binary op to compute (ADD, SUB, MUL or DIV)
 * @tparam is_fp32_dest_acc_en: enables FP32 DEST accumulation (skips bf16 RNE for DIV, ADD, SUB)
 * @tparam dst_rounding_mode: bf16 narrowing applied to ADD/SUB results (no-op if is_fp32_dest_acc_en).
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
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV,
        "calculate_sfpu_binary only supports ADD, SUB, MUL and DIV");
    static_assert(
        dst_rounding_mode == DstRoundingMode::Default || BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB,
        "NearestEven rounding parameter is currently supported for ADD and SUB only");
    // ADD/SUB round to bf16 in software only under NearestEven on a 16-bit Dest; MUL never does
    // (no narrowing), DIV always does on a 16-bit Dest and is excluded by use_two_row_binary.
    constexpr bool round_to_bf16_rne = !is_fp32_dest_acc_en && dst_rounding_mode == DstRoundingMode::NearestEven;
    if constexpr (use_two_row_binary<BINOP, round_to_bf16_rne, ITERATIONS>) {
        calculate_sfpu_binary_two_row<BINOP, ITERATIONS, TILE_SHAPE>(dst_index_in0, dst_index_in1, dst_index_out);
        return;
    }
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (BINOP == BinaryOp::ADD) {
            using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
            using Output = SfpuOperand<SfpuReg::Dest, DestBf16RneFormat<round_to_bf16_rne>>;
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

            if constexpr (BINOP == BinaryOp::SUB && round_to_bf16_rne) {
                result = float32_to_bf16_rne(result);
            }

            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        }
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
