// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// ============================================================================
// Constants for 32x32 Tile Layout
// ============================================================================
// Each face is 16 rows, tile has 4 faces arranged as:
// Face 0 (rows 0-15)  | Face 1 (rows 0-15)
// Face 2 (rows 16-31) | Face 3 (rows 16-31)

constexpr std::uint32_t NUM_FACES     = 4;
constexpr std::uint32_t ROWS_PER_LOAD = 4;

// Constants for averaging (division by 32)
constexpr std::uint32_t AVG_SHIFT_AMOUNT = 5;     // 2^5 = 32
constexpr std::uint32_t AVG_SHIFT_MASK   = 0xfff; // Mask for shift instruction encoding

// FP16B representation of 1/32 = 0.03125
constexpr std::uint32_t FP16B_ONE_OVER_32_HIGH = 0x3D00;
constexpr std::uint32_t FP16B_ONE_OVER_32_LOW  = 0x0000;

// Constants for MAX reduction
constexpr std::uint32_t ROWS_PER_TILE = 64;
constexpr std::uint32_t ROWS_PER_FACE = 16;

// ============================================================================
// Helper Functions
// ============================================================================
/**
 * @brief Apply sign magnitude conversion to four LREGs
 * Assume you loaded 4 LREGs into p_sfpu::LREG0-3 and want to convert them to sign magnitude and store in p_sfpu::LREG4-7.
 * @tparam INSTR_MOD_CAST The instruction mode for cast operations
 */
template <InstrModCast INSTR_MOD_CAST>
inline void convert_to_sign_magnitude_x4_lregs()
{
    apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INSTR_MOD_CAST);
    apply_sign_magnitude_conversion(p_sfpu::LREG1, p_sfpu::LREG5, INSTR_MOD_CAST);
    apply_sign_magnitude_conversion(p_sfpu::LREG2, p_sfpu::LREG6, INSTR_MOD_CAST);
    apply_sign_magnitude_conversion(p_sfpu::LREG3, p_sfpu::LREG7, INSTR_MOD_CAST);
}

/**
 * @brief Load data from face into LREG0-3
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param face_addr Base address of face
 * @param column_offset Column offset for the current iteration, load all rows for even columns (0) or odd columns (2) of the face
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void load_face_data(std::uint32_t face_addr, std::uint32_t column_offset)
{
    TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset);                     // rows 0-3
    TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + ROWS_PER_LOAD);     // rows 4-7
    TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 2 * ROWS_PER_LOAD); // rows 8-11
    TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 3 * ROWS_PER_LOAD); // rows 12-15
}

/**
 * @brief Load data from upper and lower faces into LREG0-7
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param upper_face_addr Base address of upper face (Face 0 or Face 1)
 * @param lower_face_addr Base address of lower face (Face 2 or Face 3)
 * @param column_offset Column offset for the current iteration
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void load_face_data(std::uint32_t upper_face_addr, std::uint32_t lower_face_addr, std::uint32_t column_offset)
{
    // Load upper face data (Face 0 or Face 1) into LREG0-3
    TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset);                     // rows 0-3
    TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + ROWS_PER_LOAD);     // rows 4-7
    TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + 2 * ROWS_PER_LOAD); // rows 8-11
    TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + 3 * ROWS_PER_LOAD); // rows 12-15

    // Load lower face data (Face 2 or Face 3) into LREG4-7
    TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset);                     // rows 0-3
    TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + ROWS_PER_LOAD);     // rows 4-7
    TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD); // rows 8-11
    TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + 3 * ROWS_PER_LOAD); // rows 12-15
}

/**
 * @brief Perform column-wise summation using transpose and replay buffer
 *
 * Process column sums for both upper and lower face using transpose and replay buffer.
 * @tparam replay_buffer_length The replay buffer index to use (6 for both int and float on Blackhole)
 */
template <std::uint32_t replay_buffer_length>
inline void sum_columns()
{
    TTI_SFPTRANSP(0, 0, 0, 0);             // Transpose: LREG0-3 → lanes 0-3, LREG4-7 → lanes 0-3 (overlapping)
    lltt::replay(0, replay_buffer_length); // Column-wise sum within each lreg after transpose
    TTI_SFPTRANSP(0, 0, 0, 0);             // Transpose back to original register layout
    lltt::replay(0, replay_buffer_length); // Sum column sums within each face after transpose
}

/**
 * @brief Perform integer averaging with proper handling of negative numbers
 * @tparam INSTRUCTION_MODE The instruction mode (determines signed vs unsigned)
 *
 * For integer formats, we need to handle negative numbers properly for division by 32.
 * Since Blackhole only supports logical shift (not arithmetic), we need to:
 * 1. Check if the number is negative using condition codes (only for signed formats)
 * 2. If negative, negate it, shift right by 5 bits, then negate back
 * 3. If positive, just shift right by 5 bits
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void perform_int_average()
{
    if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32)
    {
        // For signed Int32 format, use absolute value approach for proper division by 32
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                                      // Save original value for sign check
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);                                      // Get absolute value of LREG0
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01); // Perform logical right shift by 5 bits (divide by 32)

        // Restore sign if original value was negative
        // Check if original value was negative (sign bit set)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);               // Set condition code if original sign bit is 0 (positive)
        TTI_SFPCOMPC(0, 0, 0, 0);                           // Invert condition code (now true if original was negative)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6); // Negate LREG0 if condition is true
        TTI_SFPENCC(0, 0, 0, 0);                            // Clear condition codes
    }
    else
    {
        // For unsigned formats (UInt32), just use logical shift directly since they can't be negative
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    }
}

/**
 * @brief Perform floating-point averaging (multiply by 1/32)
 *
 * For a 32x32 tile, each column sum represents the sum of exactly 32 values (one per row).
 * This function divides by 32 by multiplying by the constant 1/32 (0.03125).
 */
inline void perform_float_average()
{
    // Load 1/32 constant (0.03125) into LREG1 for float division
    TTI_SFPLOADI(p_sfpu::LREG1, 8, FP16B_ONE_OVER_32_HIGH); // Load 0.03125 as FP16B high part
    TTI_SFPLOADI(p_sfpu::LREG1, 10, FP16B_ONE_OVER_32_LOW); // Load 0.03125 as FP16B low part

    // Multiply by 1/32 (divide by 32) - works for both float and integer formats
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
}

template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE>
inline void perform_reduce_col_sum_avg()
{
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};   // Face 0, 0, 1, 1
    constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48}; // Face 2, 2, 3, 3
    constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES]   = {0, 2, 0, 2};     // even, odd, even, odd
    // Optimized approach: Process 4 iterations to handle all column combinations
    // This reduces operations by processing complementary face pairs simultaneously, less load/store operations
    for (std::uint32_t i = 0; i < NUM_FACES; i++)
    {
        // Iteration mapping - Process vertically aligned faces (0+2, 1+3) to optimize column operations:
        // i=0: even columns, left half  (faces 0 + 2, columns 0,2,4,6,8,10,12,14)
        // i=1: odd columns,  left half  (faces 0 + 2, columns 1,3,5,7,9,11,13,15)
        // i=2: even columns, right half (faces 1 + 3, columns 16,18,20,22,24,26,28,30)
        // i=3: odd columns,  right half (faces 1 + 3, columns 17,19,21,23,25,27,29,31)
        // Key optimization: Process faces 0+2 and 1+3 (vertically aligned) instead of 0+1 and 2+3
        // This allows processing all 32 rows of a column at once (16 from upper face + 16 from lower face)
        // Reduces load/store operations by accumulating all rows into one LREG per column group
        // Final result stored in top row of upper face (first row in dest) - no intermediate storage needed
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset   = COLUMN_OFFSETS[i];
        load_face_data<INSTRUCTION_MODE>(upper_face_addr, lower_face_addr, column_offset);
        // Perform column-wise summation (Blackhole uses replay buffer 6 for both int and float)
        sum_columns<6>();
        if constexpr (is_integer_mode)
        {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4); // LREG0 = upper_face_sums + lower_face_sums (int)
        }
        else
        {
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0); // LREG0 = upper + lower (float)
        }
        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == AVG)
        {
            if constexpr (is_integer_mode)
            {
                perform_int_average<INSTRUCTION_MODE>();
            }
            else
            {
                perform_float_average();
            }
        }
        // Store the final combined column sums
        TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset);
    }
}

/**
 * @brief Performs two horizontal reductions in parallel (LREG0/LREG1 and LREG4/LREG5), interleaving
 *        instructions to hide SFPSHFT2 latency.
 *
 * SFPU hardware operates on 8 column slices in parallel but independently; column slices cannot
 * directly communicate. SFPSHFT2 is the only instruction that moves data across columns.
 * This function reduces 8 partial sums (one per column) in each of two LREG pairs down to a single
 * total sum in column 0 of each result (LREG0 and LREG4).
 *
 * Interleaving: SFPSHFT2 has 2-cycle latency and would normally require SFPNOP after each use.
 * We run two reductions in lockstep (LREG0/LREG1 and LREG4/LREG5) so that the second SFPSHFT2
 * fills the latency slot of the first, avoiding extra NOPs and reducing total cycle count.
 *
 * Algorithm (log2(8) = 3 reduction stages + 1 rotate). Each stage: copy to temp, shift right
 * by the appropriate amount so columns align, then add (SFPIADD or SFPADD) to fold halves together.
 *
 *   Phase 1: Shift by 4 and add -> 8 partial sums (cols 0-7) become 4 sums (cols 4-7).
 *   Phase 2: Shift by 2 and add -> 4 sums become 2 sums (cols 6-7).
 *   Phase 3: Shift by 1 and add -> 2 sums become 1 sum (col 7).
 *   Phase 4: Rotate right by 1  -> move the single sum from col 7 to col 0.
 *
 * @tparam is_integer_mode True for integer types (uses SFPIADD), false for float (uses SFPADD)
 */
template <bool is_integer_mode>
inline void horizontal_reduce()
{
    // Phase 1: Shift by 4 and add -> 8 partial sums (cols 0-7) become 4 sums (cols 4-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);

    // Four right-shifts-by-4 in lockstep for both pairs; second SFPSHFT2 hides first's latency
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);

    if constexpr (is_integer_mode)
    {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    }
    else
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0); // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0); // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }

    // Phase 2: Shift by 2 and add -> 4 sums become 2 sums (cols 6-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode)
    {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    }
    else
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }

    // Phase 3: Shift by 1 and add -> 2 sums become 1 sum (col 7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode)
    {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    }
    else
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0); // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0); // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }

    // Phase 4: Rotate right by 1 -> move single sum from col 7 to col 0 for store.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
    // LREG0[0 column slice] = sum of all elements in the first 4 rows of this 8-row block (first half)
    // LREG4[0 column slice] = sum of all elements in the next 4 rows of this 8-row block (second half)
}

template <InstrModLoadStore INSTRUCTION_MODE>
inline void perform_reduce_row_sum_tile(std::uint32_t tile_row_offset)
{
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Process tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31
    // Each face-pair iteration processes 8 rows (two groups of 4 rows each)
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++)
    {
        // Base offset for this face pair:
        // face_pair 0: faces 0+1 (dest indices 0-31)
        // face_pair 1: faces 2+3 (dest indices 32-63)
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++)
        {
            // Within each face, process rows in groups of 8 (two sub-groups of 4)
            std::uint32_t row_offset_first  = row_group * 8;        // 0 or 8
            std::uint32_t row_offset_second = row_offset_first + 4; // 4 or 12

            // Load 4 rows from face 0 (or 2) and face 1 (or 3)
            TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first + 2);
            TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            // Load next 4 rows from face 0 (or 2) and face 1 (or 3)
            TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
            TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second + 2);
            TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Perform vertical sum of loaded rows via replay buffer
            // After this: LREG0 contains sum of first 4 rows, LREG4 contains sum of next 4 rows
            lltt::replay(0, 6);

            // Horizontal reduction: consolidate all 8 SFPU columns into column 0 (interleaved for latency hiding)
            horizontal_reduce<is_integer_mode>();

            TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Accumulates partial row sums from all tiles in a row of tiles in tensor into tile 0 of that row.
 *
 * After per-tile row reduction, each tile has partial row sums in its column 0.
 * This function accumulates those row sums across all tiles into tile_row_base (first tile in this row of tiles in tensor).
 * Each tile already has 8 partial row-sum results in column 0 (written by perform_reduce_row_sum_tile):
 * 2 face-pairs × 2 row-groups × 2 sums per group (first 4 rows and next 4 rows of each 8-row group) = 8.
 * They are stored at row offsets 0, 4, 8, 12, 32, 36, 40, 44. Each result occupies 4 rows (one LREG).
 *
 * We process these 8 results in two batches of 4. For each batch:
 * - Load tile 0's four LREGs into LREG0-3 from the first batch's offsets.
 * - For each other tile, load its four LREGs at the same offsets into LREG4-7 and add into LREG0-3.
 * - Store LREG0-3 back to tile 0.
 * LREG4-7 hold the other tiles' data so loads and adds can be pipelined without NOPs.
 *
 * @tparam INSTRUCTION_MODE The load/store instruction mode
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void sum_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim)
{
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Row offset for each of the 8 partial row-sum results (face 0: 0,4,8,12; face 2: 32,36,40,44)
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++)
    {
        std::uint32_t base_idx = batch * 4;

        // Load tile 0's four LREGs at this batch's offsets (0,4,8,12 or 32,36,40,44) into LREG0-3
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        // Accumulate from remaining tiles
        for (std::uint32_t t = 1; t < block_ct_dim; t++)
        {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            // Load tile t's four LREGs at the same offsets into LREG4-7
            TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

            // Add LREG4-7 into LREG0-3
            if constexpr (is_integer_mode)
            {
                TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
                TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG1, 4);
                TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG2, 4);
                TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 4);
            }
            else
            {
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
                TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
                TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
            }
        }

        // Store LREG0-3 back to tile 0
        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

template <InstrModLoadStore INSTRUCTION_MODE>
inline void perform_reduce_row_sum(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim)
{
    for (std::uint32_t i = 0; i < block_rt_dim; i++)
    {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        // Step 1: Reduce each tile individually (horizontal reduction within each tile)
        for (std::uint32_t j = 0; j < block_ct_dim; j++)
        {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_sum_tile<INSTRUCTION_MODE>(tile_offset);
        }

        // Step 2: Sum column 0 from all tiles in this row into tile 0's column 0
        if (block_ct_dim > 1)
        {
            sum_first_columns_across_tiles<INSTRUCTION_MODE>(tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Runtime validation helper for supported data formats for reduce sfpu kernel
 */
constexpr bool is_supported_reduce_format(DataFormat format)
{
    return format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32 || format == DataFormat::Float16_b ||
           format == DataFormat::UInt16;
}

/**
 * @brief Configure address mode for SFPU reduce Max/Min kernel.
 * @param num_cols The number of columns in the tensor block of multiple tiles
 * @note One tile is 64 rows in dest
 */
inline void configure_addrmod_max_min(std::uint32_t num_cols)
{
    // Reduction done on first tile before looping through the rest, so we look at num_cols - 1 tile
    std::uint32_t skip_rows = (num_cols - 1) * ROWS_PER_TILE;

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = static_cast<std::int16_t>(skip_rows)},
    }
        .set(ADDR_MOD_5);
}

// ============================================================================
// Init Reduce Kernels
// ============================================================================

/**
 * @brief Initialization for SFPU reduce MAX/MIN kernel on 32x32 tile for Int32 format. Due to RTL bug INT32_2S_COMP LOAD/STORE has no effect.
 * Must cast to INT_SIGN_MAGN_TO_INT32_2S_COMP before swapping. Since CAST and SWAP are both SIMPLE instructions, cannot be integrated together in LOADMACRO
 * sequence. Therefore, we need to initialize the kernel with manual loads and stores in order to perform the CAST and SWAP operations.
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT (FP32, FP16B)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min_int32()
{
    // Initialize SFPU config and set swap direction
    _init_sfpu_config_reg();
    if constexpr (pool_type == PoolType::MAX)
    {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100); // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000); // Load upper 16 bits
        TTI_SFPCONFIG(0, 0xF, 0);
    }

    lltt::record(0, 3);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);
}

/**
 * @brief Initialization for SFPU reduce MAX/MIN kernel.
 *        Sets up LOADMACRO sequences for compare-and-swap operations, configures address modifiers,
 *        and records replay buffers for efficient column-wise maximum/minimum reduction.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT (FP32, FP16B)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @param num_cols The number of columns to process (typically 32 for a single tile, or multiple of 32 for block operations)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min(std::uint32_t num_cols)
{
    // Initialize SFPU config and set swap direction before defining LOADMACRO sequences
    _init_sfpu_config_reg();

    // Invert swap direction for MIN operations, set 8th bit in SFPU config register
    if constexpr (pool_type == PoolType::MIN)
    {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100); // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000); // Load upper 16 bits
        TTI_SFPCONFIG(0, 0xF, 0);
    }

    // Setup LOADMACRO sequence 0
    TTI_SFPSWAP(0, p_sfpu::LREG4, (0xC | p_sfpu::LREG0), 1);
    TTI_SFPLOADI(0, 0xA, 0x0084);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 4, 0);

    // Setup LOADMACRO sequence 1
    TTI_SFPSWAP(0, p_sfpu::LREG5, (0xD | p_sfpu::LREG4), 1);
    TTI_SFPLOADI(0, 0xA, 0x0085);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);

    configure_addrmod_max_min(num_cols);

    // Record replay buffer for compare-and-swap operations
    lltt::record<lltt::NoExec>(0, 11);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_SFPLOADMACRO(5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, 18);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG1, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG0, 1);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);

    // Dummy loads to increment dest counters
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_6, 0);
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_5, 0);
}

/**
 * @brief Initialization for SFPU reduce SUM and AVG kernels.
 *        Records replay buffers for column-wise summation using tree reduction.
 *        Integer modes use 6 instructions (SFPIADD), float modes use 6 (SFPADD without NOPs for Blackhole).
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT (FP32, FP16B)
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void init_reduce_sum_avg()
{
    _init_sfpu_config_reg();

    // Determine if integer or float mode based on INSTRUCTION_MODE
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Program optimized replay buffer for column summation
    // This replay buffer is called twice per iteration:
    // 1st call: After first transpose - operates on transposed data where LREG0-3 and LREG4-7 both map to lanes 0→3
    // 2nd call: After second transpose - operates on data transposed back to original layout

    if constexpr (is_integer_mode)
    {
        // Integer replay buffer: 6 instructions for column summation
        lltt::record(0, 6);

        // Upper and lower face column summation, interleaved to eliminate read-after-write dependencies.
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4); // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4); // LREG6 = LREG6 + LREG7
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4); // LREG1 = LREG1 + LREG2

        // Lower face column summation (LREG4-7)
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4); // LREG5 = LREG5 + LREG6
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // LREG0 = LREG0 + LREG1
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4); // LREG4 = LREG4 + LREG5
    }
    else
    {
        // Float replay buffer: 6 instructions (no NOPs needed for Blackhole)
        lltt::record(0, 6);

        // Upper and lower face summation chains, interleaved to eliminate read-after-write dependencies and the need for NOPs.
        // Step 1 of each chain
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0); // A1
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0); // B1
        // Step 2 of each chain
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0); // A2
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0); // B2
        // Step 3 of each chain
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0); // A3
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0); // B3
    }
}

// ============================================================================
// Calculate Functions
// ============================================================================

/**
 * @brief Column-wise maximum/minimum reduction kernel for SFPU reduce MAX/MIN operation on 32x32 tile for Int32 format. Due to RTL bug INT32_2S_COMP LOAD/STORE
 * has no effect. Must cast to INT_SIGN_MAGN_TO_INT32_2S_COMP before swapping. Since CAST and SWAP are both SIMPLE instructions, cannot be integrated together
 * in SAME LOADMACRO sequence. Therefore, we need to calculate the kernel with manual loads and stores in order to perform the CAST and SWAP operations after
 * loads.
 * @tparam INSTRUCTION_MODE The instruction mode (INT32_2S_COMP for Int32 with MAX/MIN)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type, ReduceDim reduce_dim>
inline void calculate_reduce_max_min_int32()
{
    constexpr auto INSTR_MOD_CAST             = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;
    constexpr std::uint32_t ODD_COLUMNS       = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2}; // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4]  = {
        {0, 0, 32, 32},  // j=0: Face 0 and Face 2
        {16, 16, 48, 48} // j=1: Face 1 and Face 3
    };
    constexpr std::uint32_t FINAL_REDUCE_ADDRS[2][2] = {
        {0, 32}, // j=0: Face 0 and Face 2
        {16, 48} // j=1: Face 1 and Face 3
    };

    for (std::uint32_t j = 0; j < 2; j++)
    {
        std::uint32_t top_face_addr    = FINAL_REDUCE_ADDRS[j][0]; // face 0 & 1 dst indices
        std::uint32_t bottom_face_addr = FINAL_REDUCE_ADDRS[j][1]; // face 2 & 3 dst indices

        // After this loop executes vertically adjacent faces (f0 & f2 first iteration of outer loop, f1 & f3 second iteration) are processed and store max/min
        // values in top 4 rows of their faces
        for (std::uint32_t i = 0; i < NUM_FACES; i++)
        {
            load_face_data<INSTRUCTION_MODE>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);
            convert_to_sign_magnitude_x4_lregs<INSTR_MOD_CAST>();
            lltt::replay(0, 3);

            apply_sign_magnitude_conversion(
                p_sfpu::LREG4, p_sfpu::LREG0, INSTR_MOD_CAST); // Convert back from two's complement to sign-magnitude before storing
            TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, FACE_ADDRS[j][i] + COLUMN_OFFSETS[i]);
        }

        // Load data from top 4 rows of processed vertically adjacent faces into LREG0-3
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr);
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr);
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr + ODD_COLUMNS);
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr + ODD_COLUMNS);

        convert_to_sign_magnitude_x4_lregs<INSTR_MOD_CAST>(); // Store lregs 0-3 in sign-magnitude format in lregs 4-7

        // Transpose to find absolutes max/min of top 4 rows of each face
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(0, 3);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Swap to find max/min of vertically adjacent faces
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1); // Max/Min of odd columns of face 0 and 2 (first iteration) or face 1 and 3 (second iteration)
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1); // Max/Min of even columns of face 0 and 2 (first iteration) or face 1 and 3 (second iteration)

        // Convert back from two's complement to sign-magnitude before storing
        apply_sign_magnitude_conversion(p_sfpu::LREG4, p_sfpu::LREG0, INSTR_MOD_CAST);
        apply_sign_magnitude_conversion(p_sfpu::LREG6, p_sfpu::LREG1, INSTR_MOD_CAST);

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr + ODD_COLUMNS);
    }
}

/**
 * @brief Column-wise maximum/minimum reduction kernel for SFPU reduce MAX/MIN operation.
 *        Processes a block of tiles vertically (block_height tiles stacked) and computes the maximum or minimum value
 *        for each of the columns across all rows in the block. The maximum/minimum values are placed into
 *        the first row of the output tile (row 0 of faces 0 and 1) in tilized format for each tile in the top row of tiles in the block.
 *
 *        Algorithm:
 *        - Initializes LREG4-7 with the first face pair's data (even/odd columns from faces 0 and 1)
 *        - For each tile in the block, performs compare-and-swap operations using replay buffers to find maxima/minima
 *        - Uses SFPSWAP instruction for comparisons to determine maximum/minimum between two lregs storing column data
 *        - Transposes and sorts results to align maxima/minima correctly across LREG4-7
 *        - Stores final maximum/minimum values to row 0 (32 datums across faces 0 and 1)
 *
 * @tparam pool_type The pool type (MAX or MIN)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT (FP32, FP16B)
 * @param block_height The number of tiles in the vertical block to reduce (default is 1 for single tile).
 *                     For example, block_height=4 means reduce across 4 vertically stacked tiles (128 rows total).
 */
template <PoolType pool_type, ReduceDim reduce_dim, InstrModLoadStore INSTRUCTION_MODE>
inline void calculate_reduce_max_min(const std::uint32_t block_height)
{
    static_assert(reduce_dim == REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(pool_type == PoolType::MAX || pool_type == PoolType::MIN, "Only MAX and MIN pool types are supported for this function");

    constexpr std::uint32_t replay_buffer_offset    = 9;
    constexpr std::uint32_t replay_buffer_next_face = 10;

    // Initial loads: LREG4-7 will hold maximum values across F0 and F1
    TTI_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, 18);

    // First tile processing (F0, F1, F2, F3)
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_next_face);

    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_next_face + 1);

    // Remaining tiles
    for (std::uint32_t i = 0; i < block_height - 1; i++)
    {
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face);

        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face + 1);
    }

    // Reset dest RWC counter
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Finalize: Sort and store maximum/minimum values to row 0
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG6 /*lreg_src_c*/, p_sfpu::LREG7 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG5 /*lreg_src_c*/, p_sfpu::LREG6 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG4 /*lreg_src_c*/, p_sfpu::LREG5 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store results to first row
    TTI_SFPSTORE(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, 18);
}

/**
 * @brief Column-wise sum/average reduction kernel for SFPU reduce SUM and AVG operations.
 *        Computes the sum or average of each column, placing the 32 output values into the first row
 *        of the output tile (row 0 of faces 0 and 1).
 *
 *        Uses a 4-iteration approach that processes vertically aligned face pairs (0+2, 1+3) to optimize
 *        column operations and minimize load/store operations. Each iteration handles 8 columns using
 *        transpose operations and replay buffers for tree reduction.
 *
 *        For AVG mode: Integer formats use arithmetic shift with condition codes to handle negative numbers;
 *        float formats multiply by 1/32 constant.
 *
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT (FP32, FP16B)
 */
template <PoolType pool_type, ReduceDim reduce_dim, InstrModLoadStore INSTRUCTION_MODE>
inline void calculate_reduce_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim)
{
    // Compile-time assertions to restrict to currently supported operations
    static_assert(
        reduce_dim == REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == REDUCE_ROW),
        "Only column reduction (REDUCE_COL) is supported, except row reduction (REDUCE_ROW) is allowed only for SUM");
    static_assert(pool_type == SUM || pool_type == AVG, "Only SUM and AVG pool types are currently supported on SFPU");

    // Supported instruction modes for SFPU reduce sum/avg (integer and float)
    constexpr bool is_supported_reduce_instr_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::LO16 ||
         INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::FP32 || INSTRUCTION_MODE == InstrModLoadStore::FP16B);
    static_assert(is_supported_reduce_instr_mode, "INSTRUCTION_MODE must be one of: INT32, INT32_2S_COMP, LO16, DEFAULT, FP32, FP16B");

    if constexpr (reduce_dim == REDUCE_COL)
    {
        perform_reduce_col_sum_avg<pool_type, INSTRUCTION_MODE>();
    }
    else
    {
        static_assert(pool_type == PoolType::SUM, "Row reduction (REDUCE_ROW) is allowed only for SUM");
        perform_reduce_row_sum<INSTRUCTION_MODE>(block_ct_dim, block_rt_dim);
    }
    // For column reductions: sums are stored horizontally in the first row of tensor in dest reg
    // For row reductions: sums are stored vertically in the first column of tensor in dest reg
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Unified reduction init kernel wrapper for SFPU reduce kernel.
 *        Determines the instruction mode from format, then dispatches to the appropriate init kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam format The data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b)
 * @param block_ct_dim Block dimension (used for MAX/MIN reduction to specify number of columns, default is 1 for single tile)
 */
template <PoolType pool_type, DataFormat format>
inline void _init_reduce_(std::uint32_t block_ct_dim = 1)
{
    static_assert(is_supported_reduce_format(format), "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs; Int32 MAX/MIN use INT32_2S_COMP for SFPSWAP
    // Float16_b uses DEFAULT instruction mode due to accuracy issues when RISCV_DEBUG_REG_DBG_FEATURE_DISABLE bit 11 is reset
    constexpr InstrModLoadStore INSTRUCTION_MODE = (format == DataFormat::Int32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN))
                                                       ? InstrModLoadStore::INT32_2S_COMP
                                                   : (format == DataFormat::Float16_b) ? InstrModLoadStore::DEFAULT
                                                                                       : GetSfpLoadStoreInstrMod<format>();

    // Dispatch to appropriate PoolType init
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN)
    {
        if constexpr (format == DataFormat::Int32)
        {
            init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
        }
        else
        {
            init_reduce_max_min<INSTRUCTION_MODE, pool_type>(block_ct_dim);
        }
    }
    else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG)
    {
        init_reduce_sum_avg<INSTRUCTION_MODE>();
    }
    else
    {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX || pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

/**
 * @brief Unified reduction kernel wrapper for a 32x32 tile.
 *        Determines the instruction mode from format, then dispatches to the appropriate reduction kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam format The data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b)
 * @param block_ct_dim Block dimension (used for SUM/AVG column reduction to specify number of columns, default is 1 for single tile)
 * @param block_rt_dim Block dimension (used for MAX/MIN reduction to specify block height, or SUM row reduction; default is 1 for single tile)
 *
 * @note Constraints (unable to static assert for block_rt_dim runtime parameter)
 *       - MAX/MIN with Int32 format only supports block_rt_dim == 1 (single tile)
 */
template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void _calculate_reduce_(std::uint32_t block_ct_dim = 1, std::uint32_t block_rt_dim = 1)
{
    static_assert(
        reduce_dim == REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == REDUCE_ROW),
        "Only column reduction (REDUCE_COL) is supported, except row reduction (REDUCE_ROW) is allowed only for SUM");
    static_assert(is_supported_reduce_format(format), "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs; Int32 MAX/MIN use INT32_2S_COMP for SFPSWAP
    // Float16_b uses DEFAULT instruction mode due to accuracy issues when RISCV_DEBUG_REG_DBG_FEATURE_DISABLE bit 11 is reset
    constexpr InstrModLoadStore INSTRUCTION_MODE = (format == DataFormat::Int32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN))
                                                       ? InstrModLoadStore::INT32_2S_COMP
                                                   : (format == DataFormat::Float16_b) ? InstrModLoadStore::DEFAULT
                                                                                       : GetSfpLoadStoreInstrMod<format>();

    // Dispatch to appropriate reduction kernel based on PoolType
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN)
    {
        if constexpr (format == DataFormat::Int32)
        {
            calculate_reduce_max_min_int32<INSTRUCTION_MODE, pool_type, reduce_dim>();
        }
        else
        {
            calculate_reduce_max_min<pool_type, reduce_dim, INSTRUCTION_MODE>(block_rt_dim);
        }
    }
    else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG)
    {
        calculate_reduce_sum_avg<pool_type, reduce_dim, INSTRUCTION_MODE>(block_ct_dim, block_rt_dim);
    }
    else
    {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX || pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

} // namespace sfpu
} // namespace ckernel
