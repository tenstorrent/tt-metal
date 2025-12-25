// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

/**
 * @brief Configures address modification modes for row packing.
 *
 * Sets up two address modification modes that control how the packer moves through
 * the source data during the packing operation:
 * - ADDR_MOD_0: Increments Y source by 1 (moves to next row)
 * - ADDR_MOD_1: Clears Y counter, resets to start (used at end of inner loop and to close operation)
 */
inline void _llk_pack_rows_configure_addrmod_()
{
    ckernel::addr_mod_pack_t {
        .y_src = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    ckernel::addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_1);
}

/**
 * @brief Configures the Micro-Operation (MOP) template for packing rows.
 *
 * @param num_rows Number of rows to pack from the destination register
 *
 * Creates a MOP template that controls the packing loop structure:
 * - Uses single packer interface (packer 0)
 * - Inner loop iterates num_rows times, packing one row per iteration
 * - Each iteration uses ADDR_MOD_0 to move to the next row
 * - Last inner loop instruction uses ADDR_MOD_1 to reset Y counter
 * - Outer loop runs once (no tile repetition within this operation)
 */
inline void _llk_pack_rows_mop_config_(const std::uint32_t num_rows)
{
    constexpr std::uint32_t PACKCNT          = 1; // Use only packer 0
    constexpr std::uint32_t MEGAROW          = 0; // Not using megarow mode
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t MOP_OUTER_LOOP   = 1;
    const std::uint32_t MOP_INNER_LOOP       = num_rows;

    ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));

    tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));

    tmp.program();
}

/**
 * @brief Initializes the pack rows operation.
 *
 * @param num_rows Total number of rows to pack from the destination register to L1
 *
 * This function prepares the packer hardware to pack a specified number of rows of
 * row-major data from the destination register to L1 memory. Each row contains 16 datums.
 *
 * Steps performed:
 * 1. Configures address modification modes (how to traverse data from destination register)
 * 2. Configures MOP template
 * 3. Sets up hardware counters:
 *    - X counter: Controls how many datums to pack per row
 *    - Z/W counters: Reset to zero for proper tile indexing
 */
inline void _llk_pack_rows_init_(const std::uint32_t num_rows)
{
    // In row-major layout, each row is FACE_C_DIM (16) datums.
    // A full tile has TILE_R_DIM * TILE_C_DIM / FACE_C_DIM = 32 * 32 / 16 = 64 rows.
    constexpr std::uint32_t MAX_ROWS = (TILE_R_DIM * TILE_C_DIM) / FACE_C_DIM;
    LLK_ASSERT(num_rows >= 1 && num_rows <= MAX_ROWS, "num_rows must be between 1 and 64");

    // Number of datums per row in row-major layout (16 datums = 1 row of 16 elements)
    constexpr std::uint32_t row_num_datums      = 16;
    constexpr std::uint32_t y_pos_counter_limit = 1;
    _llk_pack_rows_configure_addrmod_();
    _llk_pack_rows_mop_config_(num_rows);

    // To ensure that Y_POS counter gets reset to 0 after the operation is completed,
    // we need to set pack_reads_per_xy_plane to 1. When Y_POS counter hits that value, it will reset.
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(y_pos_counter_limit);
    // Set the packer X counter to pack the specified number of datums per row
    TTI_SETADCXX(p_setadc::PAC, row_num_datums - 1, 0x0);

    // Reset Z/W counters
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
}

/**
 * @brief Packs the specified number of rows from a destination register tile to L1 memory.
 *
 * @param tile_index Index of the tile in the destination register to read from
 * @param address L1 memory address where the packed rows will be written
 *
 * This function performs the actual row packing operation:
 * 1. Sets the W counter to the tile_index to select which dest tile to read from
 * 2. Programs the packer destination address in L1 where data will be written
 * 3. Executes the MOP template
 * 4. Closes the pack operation with ADDR_MOD_1 to clean up and reset state
 *
 * The number of rows packed was configured earlier in _llk_pack_rows_init_.
 * Each row contains 16 datums as specified by the X counter setup.
 *
 * @note This function is typically called in a loop for each tile to be packed
 */
inline void _llk_pack_rows_(const std::uint32_t tile_index, const std::uint32_t address)
{
    // Set the tile index in dest to read from
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);

    ckernel::packer::program_packer_destination(address);

    ckernel::ckernel_template::run();

    // Close the pack operation
    TTI_PACR(ADDR_MOD_1, 0, 0xf, 0, 0, 1, 1);
}

/**
 * @brief Restore packer addrmods and counters to a safe default state.
 *
 */
inline void _llk_pack_rows_uninit_()
{
    TTI_SETADCXX(p_setadc::PAC, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}
