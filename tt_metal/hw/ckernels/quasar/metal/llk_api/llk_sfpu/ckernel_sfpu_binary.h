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
#include "ckernel_sfpu_srcs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Dest-only compatibility wrapper for binary SFPU ADD, SUB, MUL and DIV.
 *
 * Arguments are Dest tile indices. This wrapper advances the hardware Dest cursor.
 * ADD delegates to calculate_add_operands; callers with independently located operands
 * can call that core directly. SUB, MUL and DIV are not yet on the operand model.
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
    bool APPROXIMATION_MODE /*maybe_unused*/,
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
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (BINOP == BinaryOp::ADD) {
            using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
            constexpr bool round_to_bf16 = !is_fp32_dest_acc_en && dst_rounding_mode == DstRoundingMode::NearestEven;
            using Output = SfpuOperand<SfpuReg::Dest, DestBf16RneFormat<round_to_bf16>>;
            calculate_add_operands<1>(
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

            if constexpr (
                BINOP == BinaryOp::SUB && !is_fp32_dest_acc_en && dst_rounding_mode == DstRoundingMode::NearestEven) {
                result = float32_to_bf16_rne(result);
            }

            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        }
        sfpi::dst_reg++;
    }
}

// Reference adapter for the unified Dest/SrcS API; exercised by test_sfpu_add_parallel_matmul_quasar.
/**
 * @brief ADD over one SrcS slice (slots per @ref SrcsLayout).
 *
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @note The caller runs unpack/pack and clears the SrcS valids after this call, as
 *       llk_sfpu_srcs_binary does.
 */
template <sfpi::DataLayout LAYOUT>
sfpi_inline void calculate_add_srcs() {
    using Layout = SrcsLayout<LAYOUT>;
    using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
    calculate_add_operands<Layout::ops>(Operand{Layout::in0}, Operand{Layout::in1}, Operand{Layout::out});
}

/**
 * @brief Initialisation hook for binary SFPU kernels.
 * For DIV, programs the Newton-Raphson reciprocal constant; no-op for MUL.
 *
 * @tparam APPROXIMATION_MODE: forwarded to the op-specific init
 * @tparam BINOP: selects which op's init to run
 */
template <bool APPROXIMATION_MODE /*maybe_unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::DIV) {
        _init_reciprocal_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
