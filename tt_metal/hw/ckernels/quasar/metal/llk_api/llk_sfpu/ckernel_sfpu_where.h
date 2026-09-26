// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Select between floating-point operands using a floating-point condition.
 *
 * All four operands independently select Dest/SrcS, layout and base offset in SFPI
 * index units. With default SfpiFormat policies this advances only explicit indices;
 * the caller configures address modes and formats and handles synchronization.
 * Each input range must coincide with the output or be disjoint from it. All three
 * inputs are loaded before the store, allowing the output to overwrite any one input.
 * Keep offsets constant where possible to enable immediate instruction addresses.
 *
 * SrcS callers must supply three populated input ranges and arrange packing/completion.
 * The existing unary/binary SrcS pipeline wrappers do not supply a third input.
 */
template <int ITERATIONS, class Condition, class TrueInput, class FalseInput, class Output>
sfpi_inline void calculate_where_operands(
    const Condition& condition, const TrueInput& true_input, const FalseInput& false_input, const Output& output) {
    static_assert(ITERATIONS > 0, "WHERE requires at least one SFPI access");
    static_assert(
        std::is_same_v<typename Condition::value_type, sfpi::vFloat> &&
            std::is_same_v<typename TrueInput::value_type, sfpi::vFloat> &&
            std::is_same_v<typename FalseInput::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Output::value_type, sfpi::vFloat>,
        "WHERE requires floating-point operands");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat cond = condition.load(d);
        sfpi::vFloat true_val = true_input.load(d);
        sfpi::vFloat result = false_input.load(d);

        v_if(cond != 0) { result = true_val; }
        v_endif;

        output.store(d, result);
    }
}

/**
 * @brief Per-lane ternary select: @c out = (cond == 0) ? false_val : true_val.
 *
 * Loads @c false_val directly into the @c result variable so sfpi aliases
 * them to the same LREG. The @c cond==0 branch then becomes implicit — only
 * the @c cond!=0 lanes need a CC-gated move to overwrite the result reg
 * with @c true_val. sfpi emits the same sequence as the explicit
 * @c binary_comp pattern:
 *
 *     SFPLOAD cond
 *     SFPLOAD true_val
 *     SFPLOAD false_val (= result reg)
 *     SFPSETCC (cond == 0)
 *     SFPCOMPC
 *     SFPMOV result <- true_val
 *     SFPENCC
 *     SFPSTORE result
 *     TTINCRWC (from dst_reg++) — advances dest counter by one SFP row pair.
 *
 * @tparam APPROXIMATION_MODE Unused for @c where; kept for API parity with
 *         other SFPU kernels.
 * @tparam ITERATIONS         Inner SFPU row-pair count per face. Defaults
 *         to 8 for the standard 16-row face. The outer per-face loop and
 *         section base setup are owned by
 *         @c _llk_math_eltwise_ternary_sfpu_params_.
 * @tparam TILE_SHAPE         Destination tile shape used to calculate operand
 *         offsets.
 *
 * @param dst_index_in0 DEST tile index holding the condition operand.
 * @param dst_index_in1 DEST tile index holding the true-branch operand.
 * @param dst_index_in2 DEST tile index holding the false-branch operand.
 * @param dst_index_out DEST tile index that receives the per-lane result.
 */
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_where(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_where_operands<1>(
            Operand{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
            Operand{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
            Operand{static_cast<int>(dst_index_in2 * dst_tile_size_sfpi)},
            Operand{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
