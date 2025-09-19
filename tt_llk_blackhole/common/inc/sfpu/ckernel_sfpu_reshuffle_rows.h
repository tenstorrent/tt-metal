// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

/**
 * @brief Implements gradient accumulation with row reshuffling for embedding backward pass
 *
 * This function performs a scatter-add operation: output[mask[i]] += input[i] for each row i (0-31).
 * It's the core SFPU implementation for embedding gradient accumulation, where gradients from
 * different input positions are accumulated into their corresponding embedding rows based on
 * a mask that specifies the destination mapping.
 *
 * Algorithm:
 * - Input:  Gradient tile (tile 0) + destination row mask (idx_addr)
 * - Output: Accumulated gradients in reshuffled pattern (tile 1, offset 64)
 * - For each input row i: if mask[i] < 32, then output[mask[i]] += input[i]
 * - Mask value 255 indicates "skip this row" (no accumulation)
 *
 * SFPU Implementation Details:
 * - Leverages vector register parallelism for efficient row processing
 * - Uses face-aware addressing to handle tile memory layout (faces 0/1 for rows 0-15, faces 2/3 for rows 16-31)
 * - Employs transpose operations to work around SFPLOAD/SFPSTORE 4-row granularity constraints
 * - Processes both even/odd columns simultaneously using +2 offset addressing
 *
 * @param idx_addr L1 address of the mask tile containing destination row mappings (uint8_t[32])
 */
inline void _calculate_reshuffle_rows_(const uint idx_addr)
{
    constexpr uint output_tile_offset = 64;

    // clr DEST tile 1
    // TODO (Radomir): Add optional clear that is more optimal using tile copy
    // for (uint row=0; row < 32; row+=4) {
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 2);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 32);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 34);
    // }

    // Skip tile header, hence + 16:
    volatile tt_l1_ptr uint8_t *idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(idx_addr + 16);

    // TODO: Add dynamic assert for idx_ptr being within L1 memory bounds
    // using hardware memory map constants: MEM_L1_BASE and MEM_L1_SIZE

    static constexpr uint input_lreg[4]  = {p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3};
    static constexpr uint output_lreg[4] = {p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7};

    for (uint row = 0; row < 32; row++)
    {
        // Calculate base address for 4-row groups within tile faces
        // SFPU loads 4 consecutive rows at once, targeting even/odd columns with +2 offset
        // (row & ~0x3): Round down to nearest multiple of 4
        // (row & 0x10): Extract bit 4 → 0 for rows 0-15, 16 for rows 16-31
        // Row 0-3   → addr 0  (0 + 0 = 0)   Faces 0/1, group 0
        // Row 4-7   → addr 4  (4 + 0 = 4)   Faces 0/1, group 1
        // Row 8-11  → addr 8  (8 + 0 = 8)   Faces 0/1, group 2
        // Row 12-15 → addr 12 (12 + 0 = 12) Faces 0/1, group 3
        // Row 16-19 → addr 32 (16 + 16 = 32) Faces 2/3, group 0
        // Row 20-23 → addr 36 (20 + 16 = 36) Faces 2/3, group 1
        uint input_row_addr = (row & ~0x3) + (row & 0x10);
        uint input_row_lreg = input_lreg[row % 4];

        uint dst_row = static_cast<uint>(idx_ptr[row]);
        // Skip if dst_row is 255, i.e. mask is invalid and we don't want to process the current row
        if (dst_row >= 32)
        {
            continue;
        }
        uint output_row_addr = (dst_row & ~0x3) + (dst_row & 0x10);
        uint output_row_lreg = output_lreg[dst_row % 4];

        // load in the input row and output row
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, input_row_addr);                            // Face 0/2, even columns
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, input_row_addr + 2);                        // Face 0/2, odd columns
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, input_row_addr + 16);                       // Face 1/3, even columns
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, input_row_addr + 18);                       // Face 1/3, odd columns
        TT_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, output_tile_offset + output_row_addr);      // Face 0/2, even columns
        TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 2);  // Face 0/2, odd columns
        TT_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 16); // Face 1/3, even columns
        TT_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 18); // Face 1/3, odd columns
        // TRANSPOSE #1: Rearrange loaded 4-row blocks to isolate target rows
        // SFPLOAD loads 4 consecutive rows (e.g., rows 4-7) into LREG0-3, but we only want one specific row (e.g., row 5)
        // This transpose shuffles the register contents so row 5 data becomes accessible via input_row_lreg[1]
        TTI_SFPTRANSP(0, 0, 0, 0); // Puts desired input row into LREG "input_row_lreg" and output row into "output_row_lreg"

        // ACCUMULATION: Perform gradient accumulation for embedding backward pass
        // Implements: output[dst_row] += input[row] (scatter-add operation)
        // Uses LCONST_1 (value 1.0) as multiplier: dst = 1.0 * src + dst
        TT_SFPADD(input_row_lreg, p_sfpu::LCONST_1, output_row_lreg, output_row_lreg, 0);

        // TRANSPOSE #2: Rearrange accumulated results back to 4-row storage format
        // Prepares the computed result for SFPSTORE, which expects data in LREG4-7 positions
        // This undoes the first transpose to match the expected storage layout
        TTI_SFPTRANSP(0, 0, 0, 0);                                                            // Puts desired output row back into LREG4-7 for storage
        TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, output_tile_offset + output_row_addr);      // Face 0/2, even columns
        TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 2);  // Face 0/2, odd columns
        TT_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 16); // Face 1/3, even columns
        TT_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_7, output_tile_offset + output_row_addr + 18); // Face 1/3, odd columns
    }
}

} // namespace sfpu
} // namespace ckernel
