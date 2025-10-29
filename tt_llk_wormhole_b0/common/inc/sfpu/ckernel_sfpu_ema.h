// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "sfpi.h"

/*
 * @brief Loads the current input at row (I * 4) + J from the current tile in dst 0.
 *
 * @tparam I
 * @tparam J
 *
 * 4 inputs each from 32 columns at the current offset are loaded into the LREG0-3 registers
 * respectively from the current tile in dst 0.
 * Values are expected to be bfloat16 format.
 */
template <uint32_t I, uint32_t J>
sfpi_inline void _ema_load_current_input_()
{
    constexpr uint32_t tile_offset    = 0; // offset for tile 0 in dst
    constexpr uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr uint32_t offset0        = dst_reg_offset;
    constexpr uint32_t offset1        = dst_reg_offset + 2;
    constexpr uint32_t offset2        = dst_reg_offset + 16;
    constexpr uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset0); // row0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset1); // row1
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset2); // row2
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset3); // row3
}

/**
 * @brief Stores the current input at row (I * 4) + J to the corresponding position in tile 1.
 *
 * @tparam I
 * @tparam J
 *
 * 4 inputs each from 32 columns at the current offset are stored from the LREG0-3 registers
 * respectively to tile 1.
 * Values are stored in bfloat16 format.
 */
template <uint32_t I, uint32_t J>
sfpi_inline void _ema_store_current_input_()
{
    constexpr uint32_t tile_offset    = 64; // offset for tile 1 in dst
    constexpr uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr uint32_t offset0        = dst_reg_offset;
    constexpr uint32_t offset1        = dst_reg_offset + 2;
    constexpr uint32_t offset2        = dst_reg_offset + 16;
    constexpr uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset0); // row0
    TTI_SFPSTORE(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset1); // row1
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset2); // row2
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, ckernel::ADDR_MOD_3, offset3); // row3
}

/*
 * @brief Computes the exponential moving average for 4 inputs each from 32 columns.
 *
 * The function computes the exponential moving average for 4 inputs each from 32 columns.
 * It operates on the input data in the LREG0-3 registers, updating the EMA values using the loaded
 * alpha and beta parameters. It takes the previous EMA value from LREG4 for all 32 columns.
 * The output is stored in the LREG0-3 registers.
 */
sfpi_inline void _compute_ema_math_()
{
    // Transpose the input data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);

    // EMA equation: EMA_new = α * EMA_old + β * input
    // Where: LREG0=input, LREG4=EMA_old, LREG5=α, LREG6=(1-α), LREG7=EMA_new
    // Thus, LREG7 = LREG5 * LREG4 + LREG6 * LREG0
    // We do this for each of the 4 inputs.

    // Step 1(in0): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG4 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in0): Calculate final EMA = β * in0 + α * EMA_old
    // LREG0 = (LREG6 * LREG0) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG0 (2-cycle operation)

    // Step 1(in1): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG0 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in1): Calculate final EMA = β * in1 + α * EMA_old
    // LREG1 = (LREG6 * LREG1) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG1 (2-cycle operation)

    // Step 1(in2): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG1 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in2): Calculate final EMA = β * in2 + α * EMA_old
    // LREG2 = (LREG6 * LREG2) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG2 (2-cycle operation)

    // Step 1(in3): Calculate α * EMA_old in LREG7
    // LREG7 = LREG5 * LREG2 (α * EMA_old)
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG7 (2-cycle operation)

    // Step 2(in3): Calculate final EMA = β * in3 + α * EMA_old
    // LREG3 = (LREG6 * LREG3) + LREG7
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG3 (2-cycle operation)

    // Update EMA_old for next iteration
    // LREG4 = LREG3 (copy new EMA to old EMA register)
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);

    // Transpose the output data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);
}

/**
 * @brief Processes a single EMA block (load inputs, compute EMA, store results).
 *
 * @tparam I Row group index (0-1)
 * @tparam J Column group index (0-3)
 *
 * This is a helper function that performs all three steps for a single block:
 * load inputs, compute EMA, and store results.
 */
template <uint32_t I, uint32_t J>
sfpi_inline void _process_ema_block_()
{
    _ema_load_current_input_<I, J>();
    _compute_ema_math_();
    _ema_store_current_input_<I, J>();
}

namespace ckernel
{
namespace sfpu
{
/**
 * @brief Loads the alpha and beta values into the corresponding SFPU registers.
 *
 * @param alpha The alpha parameter, typically the smoothing factor for the EMA, in 32-bit format.
 * @param beta  The beta parameter, in 32-bit format, representing (1 - alpha) or a similar value.
 *
 * The values dictate the amount of weight given to the previous output and the current input.
 * It follows the formula: EMA_new = α * EMA_old + β * input
 * These values are loaded into the LREG5 (α) and LREG6 (β) registers.
 * The 32 bit values are expected to be the float32 representation of the alpha and beta values.
 */
sfpi_inline void _load_alpha_beta_(uint32_t alpha, uint32_t beta)
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, alpha >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, alpha & 0xFFFF);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, beta >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, beta & 0xFFFF);
}

/**
 * @brief Clears the previous EMA output stored in the designated register (LREG4).
 *
 * This function zeroes out the register (LREG4) used for storing the previous EMA value,
 * preparing for a new calculation cycle. Typically invoked at the beginning of the
 * calculation for a new EMA sequence.
 */
sfpi_inline void _clear_previous_output_()
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
}

/**
 * @brief Calculates the Exponential Moving Average (EMA) for the input tile in dst 0.
 *
 * Executes the EMA update for all input fragments in the tile in dst 0.
 * The function processes the input data in a loop, updating the EMA values using the loaded
 * alpha and beta parameters. It operates on 32 inputs each from 32 columns, present in a
 * 32x32 tile in dst 0. It also takes the previous EMA value from LREG4 for all 32 columns.
 * The output is stored in the corresponding position in tile 1. The last output for all 32
 * columns is also held in LREG4 for use by the next tile.
 */
sfpi_inline void _calculate_ema_tile_()
{
    // Transpose the input data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);

    // We load 4 rows of a tile (with 32 columns each) at a time and process them.
    // To finish the entire tile, we need to repeat this process 8 times.

    // Process the first block (4 rows of 32 columns)
    _process_ema_block_<0, 0>();

    // Repeat this 7 more times to process the remaining blocks
    _process_ema_block_<0, 1>();
    _process_ema_block_<0, 2>();
    _process_ema_block_<0, 3>();

    _process_ema_block_<1, 0>();
    _process_ema_block_<1, 1>();
    _process_ema_block_<1, 2>();
    _process_ema_block_<1, 3>();

    // Transpose the output data to the correct order
    TTI_SFPTRANSP(0, 0, 0, 0);
}
} // namespace sfpu
} // namespace ckernel
