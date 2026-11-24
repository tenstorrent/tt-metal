// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_pack_rows.h: same Tensix instruction sequence and access kinds as production.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h" // not used here: kept for production-copy headers that include llk_pack_rows.h
#include "hal/counters/adc.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack.h" // hal_port::set_addr_mod_pack
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
    using hal::cfg::Sec;
    hal_port::set_addr_mod_pack<Sec::S0, 1 /*y_src_incr*/, 0 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();
    hal_port::set_addr_mod_pack<Sec::S1, 0 /*y_src_incr*/, 1 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();
}

/**
 * @brief Configures the MOP template for packing rows.
 *
 * The MOP uses a single outer loop with num_rows inner iterations:
 * - Each inner iteration packs one row (16 datums) using ADDR_MOD_0 (Y += 1)
 * - The last outer loop iteration uses ADDR_MOD_1 with Last=1 to reset Y to 0 after num_rows rows
 *
 * @param num_rows: Number of rows to pack from the destination register.
 */
inline void _llk_pack_rows_mop_config_(const std::uint32_t num_rows)
{
    using namespace hal::mop;
    using hal::pack::DataTransfer;
    using hal::pack::TileBoundary;

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = num_rows;

    // Normal dest access, single interface, ADDR_MOD_0 (Y += 1), zero output disabled.
    constexpr std::uint32_t row = DataTransfer {.address_modifier = ADDR_MOD_0, .interfaces = p_pacr::SINGLE_INTF_ACTIVE}.get_operation();
    // Last outer loop instruction must have Last=1 to close the block
    // and use ADDR_MOD_1 to reset Y counter
    constexpr std::uint32_t last_row =
        DataTransfer {.address_modifier = ADDR_MOD_1, .interfaces = p_pacr::SINGLE_INTF_ACTIVE, .boundary = TileBoundary::Last}.get_operation();

    // Explicit NOP end ops keep production's MOP_CFG stores (end_op0/end_op1 = NOP) unchanged. Outer count is 1, so
    // the last-inner op is never issued (production leaves it equal to the loop op).
    program(loop(MOP_OUTER_LOOP, loop(MOP_INNER_LOOP, row), TT_OP_NOP, TT_OP_NOP).patch(last_outer, last_row));
}

/**
 * @brief Initializes the pack rows operation.
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
 *
 * @param num_rows: Total number of rows to pack from the destination register to L1.
 * @note Pair with @ref _llk_pack_rows_uninit_ after the matching @ref _llk_pack_rows_ execute calls.
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
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::PackCounters::pack_reads_per_xy_plane, hal::cfg::Sec::S0, y_pos_counter_limit>();

    namespace adc = hal::counters::adc;
    // Set the packer X counter to pack the specified number of datums per row: SETADCXX(PAC, row_num_datums - 1, 0)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = row_num_datums - 1}}.for_clients<adc::Client::Packers>()>();

    // Reset Z/W counters: SETADCZW(PAC, mask 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Packers>()>();
}

/**
 * @brief Packs the specified number of rows from a destination register tile to L1 memory.
 *
 * This function performs the actual row packing operation:
 * 1. Sets the W counter to the tile_index to select which dest tile to read from
 * 2. Programs the packer destination address in L1 where data will be written
 * 3. Executes the MOP template
 * 4. Reset Z counters after pack operation
 *
 * @param tile_index: Index of the tile in the destination register to read from.
 * @param address: L1 memory address where the packed rows will be written.
 * @note Call @ref _llk_pack_rows_init_ to program the row count and counters before this function, and
 *       @ref _llk_pack_rows_uninit_ once all row-pack calls are complete.
 */
inline void _llk_pack_rows_(const std::uint32_t tile_index, const std::uint32_t address)
{
    // Set the tile index in dest to read from
    set_dst_write_addr(tile_index);

    hal_port::program_packer_destination(address);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Reset Z counters after pack operation: SETADCZW(PAC, mask 0b0101)
    hal::counters::adc::set<hal::counters::adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<hal::counters::adc::Client::Packers>()>();
}

/**
 * @brief Restore packer addrmods and counters to a safe default state.
 *
 * Resets the packer X counter to its default full-face value, undoing @ref _llk_pack_rows_init_.
 *
 * @note Pairs with @ref _llk_pack_rows_init_.
 */
inline void _llk_pack_rows_uninit_()
{
}
