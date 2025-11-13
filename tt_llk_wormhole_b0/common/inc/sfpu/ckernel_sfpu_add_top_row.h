// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

/**
 * @brief Add top row operation for a 32x32 tile.
 *        Automatically chooses between integer and floating-point implementations based on the data format.
 *        Takes the top row of tile 0 (first 16 datums of face 0 and first 16 of face 1) and adds them
 *        with the top row of tile 1 (first 16 datums of face 2 and first 16 of face 3).
 * @tparam format The data format that determines which implementation to use.
 *                Supported formats:
 *                - DataFormat::Int32: Use integer implementation with INT32 instruction mode
 *                - DataFormat::UInt32: Use integer implementation with INT32_2S_COMP instruction mode
 *                - DataFormat::Float32: Uses floating-point implementation with FP32 instruction mode
 * @param tile_idx_0 The index of the first tile in the Dest register to operate on.
 * @param tile_idx_1 The index of the second tile in the Dest register to operate on.
 * @param tile_idx_dst The index of the result tile in the Dest register where the result will be stored.
 */
template <DataFormat format>
inline void _calculate_add_top_row_(const uint tile_idx_0 = 0, const uint tile_idx_1 = 0, const uint tile_idx_dst = 0)
{
    static_assert(
        format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32,
        "Unsupported data format. Supported formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::Float32");

    // Determine instruction mode and replay buffer parameters based on format
    constexpr InstrModLoadStore INSTRUCTION_MODE = (format == DataFormat::Int32)    ? InstrModLoadStore::INT32
                                                   : (format == DataFormat::UInt32) ? InstrModLoadStore::INT32_2S_COMP
                                                                                    : InstrModLoadStore::FP32;

    constexpr uint REPLAY_BUFFER_INDEX = (format == DataFormat::Float32) ? 4 : 0;
    constexpr uint REPLAY_BUFFER_COUNT = 4;

    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    const uint tile_offset_0     = tile_idx_0 * dst_tile_size;
    const uint tile_offset_1     = tile_idx_1 * dst_tile_size;
    const uint tile_offset_dst   = tile_idx_dst * dst_tile_size;

    // Load upper row (Face 0 and Face 1) of Tile 0
    TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_0);          // face 0, rows 0-3, even columns
    TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_0 + 2);      // face 0, rows 0-3, odd columns
    TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_0 + 16);     // face 1, rows 0-3, even columns
    TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_0 + 16 + 2); // face 1, rows 0-3, odd columns

    // Load upper row (Face 2 and Face 3) of Tile 1
    TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_1);          // face 2, rows 0-3, even columns
    TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_1 + 2);      // face 2, rows 0-3, odd columns
    TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_1 + 16);     // face 3, rows 0-3, even columns
    TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_1 + 16 + 2); // face 3, rows 0-3, odd columns

    // Call replay buffer (integer: index 0, 4 instructions; float: index 4, 8 instructions)
    lltt::replay(REPLAY_BUFFER_INDEX, REPLAY_BUFFER_COUNT);

    TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_dst);
    TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_dst + 2);
    TT_SFPSTORE(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_dst + 16);
    TT_SFPSTORE(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset_dst + 16 + 2);
}

/**
 * @brief Initialize SFPU configuration register and set up replay buffers for add top row kernel.
 *        Sets up two replay buffers:
 *        - Integer replay buffer (index 0, 4 instructions): TTI_SFPIADD operations
 *        - Float replay buffer (index 4, 8 instructions): TT_SFPADD + TTI_SFPNOP operations
 */
inline void _init_add_top_row_()
{
    _init_sfpu_config_reg();

    // Set up integer replay buffer (index 0, 4 instructions)
    // Contains: 4 TTI_SFPIADD operations for adding top rows
    lltt::record(0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG1, 4);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG2, 4);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 4);

    // Set up floating-point replay buffer (index 4, 8 instructions)
    // Contains: 4 TT_SFPADD + 4 TTI_SFPNOP operations for adding top rows
    lltt::record(4, 4);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
}

} // namespace sfpu
} // namespace ckernel
