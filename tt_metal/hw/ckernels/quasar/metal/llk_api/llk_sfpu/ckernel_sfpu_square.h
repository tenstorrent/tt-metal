// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Square floating-point operands with independently selected locations and formats.
 *
 * Operands supply load/store in SFPI index units. With the default SfpiFormat policy,
 * this loop advances only explicit indices; the caller owns setup and synchronization.
 * Input/output ranges must coincide or be disjoint. Keep offsets constant for immediate addresses.
 */
template <int ITERATIONS, class Input, class Output>
sfpi_inline void calculate_square_operands(const Input& input, const Output& output) {
    static_assert(ITERATIONS > 0, "Square requires at least one SFPI access");
    static_assert(
        std::is_same_v<typename Input::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Output::value_type, sfpi::vFloat>,
        "Square requires floating-point operands");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat value = input.load(d);
        output.store(d, value * value);
    }
}

/**
 * @brief Configure the SFPU address mode used by the square op.
 *
 * Programs ADDR_MOD_6 with a dest increment of 2 (one SFPU pass writes 2 rows on Quasar).
 *
 * @note Call this before @ref calculate_square to set up the address mode it relies on.
 */
inline void init_square() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Square a Dest span in place: dest = x * x (one face with default ITERATIONS).
 *
 * @tparam ITERATIONS: Number of SFPU passes (each covers 2 rows).
 * @note Call @ref init_square before this to program the address mode it depends on.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
    using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat, ADDR_MOD_6>>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // The store advances Dest; keep explicit operand indices fixed at zero.
        calculate_square_operands<1>(Input{}, Output{});
    }
}

// SrcS layout follows llk_sfpu_srcs_api.h: input at the slice base, output at
// + 2 * YDIM. UnpackSrcS selects the register file; indices are in SFPI steps.
static_assert(sfpi::SFP_SRCSREG_STRIDE == ckernel::math::SFP_ROWS, "one sfpi index step must be one SFPU op");

/**
 * @brief Calculates floating-point square over one SrcS slice.
 *
 * @tparam YDIM: rows per SrcS slice (trisc::srcs_dims::ydim).
 * @tparam IN_LAYOUT: Input sfpmem layout, also selecting the slice geometry.
 * @tparam OUT_LAYOUT: Output sfpmem layout, defaulting to IN_LAYOUT. The caller must
 *         configure PACK1 to read this format and ensure the output fits its SrcS range.
 * @note The caller configures unpack/pack and clears the SrcS valids after this call,
 *       as llk_sfpu_srcs_unary does. This kernel does not signal completion itself.
 */
template <int YDIM, sfpi::DataLayout IN_LAYOUT, sfpi::DataLayout OUT_LAYOUT = IN_LAYOUT>
sfpi_inline void calculate_square_srcs() {
    static_assert(YDIM > 0 && YDIM % ckernel::math::SFP_ROWS == 0, "SrcS slice must contain whole SFPU passes");
    constexpr int ops = YDIM / static_cast<int>(ckernel::math::SFP_ROWS);
    using Input = SfpuOperand<SfpuReg::SrcS, SfpiFormat<IN_LAYOUT, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::SrcS, SfpiFormat<OUT_LAYOUT, sfpi::vFloat>>;
    calculate_square_operands<ops>(Input{0}, Output{2 * ops});
}

/**
 * @brief Square values with independently selected input and output register spaces.
 *
 * @tparam ITERATIONS: Number of SFPI accesses; for a full SrcS slice use YDIM / SFP_ROWS.
 * @tparam IN_LAYOUT: Input load layout.
 * @tparam OUT_LAYOUT: Output store layout, defaulting to IN_LAYOUT.
 * @param input_offset: Input base in SFPI index units (one index step is two address rows).
 * @param output_offset: Output base in SFPI index units, independently of input_offset.
 *
 * Dest indices are relative to the caller's current Dest cursor. SrcS indices are
 * relative to UnpackSrcS's base; do not include SFPU_SRCS_BASE_ADDR. For the current
 * SrcS pipeline, input slot 0 starts at index 0 and output slot 2 at YDIM.
 * The caller configures ADDR_MOD_7 with zero increments, sets up the register files,
 * and handles Dest synchronization and SrcS completion. This function advances
 * only its explicit indices; it does not increment the Dest cursor or clear valids.
 * Supply constant offsets where possible to allow immediate load/store addresses.
 * Input and output ranges may coincide or be disjoint; partial overlap is unsupported.
 */
template <
    SfpuReg IN_REG,
    SfpuReg OUT_REG,
    int ITERATIONS,
    sfpi::DataLayout IN_LAYOUT = sfpi::DataLayout::Default,
    sfpi::DataLayout OUT_LAYOUT = IN_LAYOUT>
sfpi_inline void calculate_square_regs(const int input_offset, const int output_offset) {
    using Input = SfpuOperand<IN_REG, SfpiFormat<IN_LAYOUT, sfpi::vFloat>>;
    using Output = SfpuOperand<OUT_REG, SfpiFormat<OUT_LAYOUT, sfpi::vFloat>>;
    calculate_square_operands<ITERATIONS>(Input{input_offset}, Output{output_offset});
}

}  // namespace sfpu
}  // namespace ckernel
