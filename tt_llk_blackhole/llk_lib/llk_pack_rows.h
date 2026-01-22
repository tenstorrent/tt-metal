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
#include "llk_memory_checks.h"
#include "llk_pack_common.h"

/**
 * @brief Configures address modification modes for row packing.
 *
 * Address mods are applied automatically after each PACR instruction to update counters.
 * - ADDR_MOD_0: Increment Y by 1 (advance to next row in dest)
 * - ADDR_MOD_1: Clear Y to 0 (reset for next tile)
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
 * @brief Configures the MOP template for packing rows.
 *
 * @param num_rows Number of rows to pack from the destination register
 *
The MOP uses a single outer loop with num_rows inner iterations:
 * - Each inner iteration packs one row (16 datums) using ADDR_MOD_0 (Y += 1)
 * - The last outer loop iteration uses ADDR_MOD_1 with Last=1 to reset Y to 0 after num_rows rows
 */
inline void _llk_pack_rows_mop_config_(const std::uint32_t num_rows)
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t MOP_OUTER_LOOP   = 1;
    const std::uint32_t MOP_INNER_LOOP       = num_rows;

    ckernel::ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            p_pacr::SINGLE_INTF_ACTIVE,
            0, // OvrdThreadId
            0, // Concat
            p_pacr::NO_CTXT_CTRL,
            0, // Flush
            0) // Last
    );

    // Last outer loop instruction must have Last=1 to close the block
    // and use ADDR_MOD_1 to reset Y counter
    tmp.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        ZERO_OUTPUT_FLAG,
        p_pacr::SINGLE_INTF_ACTIVE,
        0, // OvrdThreadId
        0, // Concat
        p_pacr::NO_CTXT_CTRL,
        0, // Flush
        1) // Last
    );

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
 *    - Z/W counters: Reset to zero
 */
inline void _llk_pack_rows_init_(const std::uint32_t num_rows)
{
    // In row-major layout, each row is FACE_C_DIM (16) datums.
    // A full tile has TILE_R_DIM * TILE_C_DIM / FACE_C_DIM = 32 * 32 / 16 = 64 rows.
    constexpr std::uint32_t MAX_ROWS = (TILE_R_DIM * TILE_C_DIM) / FACE_C_DIM;
    LLK_ASSERT(num_rows >= 1 && num_rows <= MAX_ROWS, "num_rows must be between 1 and 64");

    constexpr std::uint32_t row_num_datums      = FACE_C_DIM;
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
 * 4. Reset Z counters after pack operation
 */
inline void _llk_pack_rows_(const std::uint32_t tile_index, const std::uint32_t address)
{
    // Set the tile index in dest to read from
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);

    ckernel::packer::program_packer_destination(address);

    ckernel::ckernel_template::run();

    // Reset Z counters after pack operation
    TT_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

/**
 * @brief Restore packer addrmods and counters to a safe default state.
 *
 */
inline void _llk_pack_rows_uninit_()
{
    // Restore X counter to default
    TTI_SETADCXX(p_setadc::PAC, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}
