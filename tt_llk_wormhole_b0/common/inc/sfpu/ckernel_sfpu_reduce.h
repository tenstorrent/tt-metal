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
 * @brief Generic reduction operation for a 32x32 tile, placing output values into the first row.
 *        Currently able to calculate column-wise sum and/or average of a 32x32 tile, placing output values into the first row.
 *        Uses an optimized approach that processes vertically aligned face pairs (0+2, 1+3) to minimize
 *        load/store operations and eliminate intermediate storage requirements.
 *        For integer formats with averaging, handles negative numbers properly using condition codes
 *        since Wormhole B0 only supports logical shift (not arithmetic shift).
 * @tparam pool_type The pool/reduction pool_type (SUM, AVG, MAX). Currently only SUM and AVG are supported.
 * @tparam reduce_dim The reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR). Currently only REDUCE_COL is supported.
 * @tparam INSTRUCTION_MODE The instruction modifier that determines the data type and precision:
 *                          - InstrModLoadStore::INT32: Signed 32-bit integers
 *                          - InstrModLoadStore::INT32_2S_COMP: 32-bit integers in 2's complement (no effect in Wormhole B0)
 *                          - InstrModLoadStore::LO16: Unsigned 16-bit integers (lower 16 bits)
 */
template <PoolType pool_type, ReduceDim reduce_dim, InstrModLoadStore INSTRUCTION_MODE>
inline void calculate_reduce_int()
{
    // Compile-time assertions to restrict to currently supported operations
    static_assert(reduce_dim == REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported on SFPU");
    static_assert(pool_type == SUM || pool_type == AVG, "Only SUM and AVG pool types are currently supported on SFPU");
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Pre-calculated face addresses and column offsets for each iteration
    // Each face is 16 rows, tile has 4 faces arranged as:
    // Face 0 (rows 0-15)  | Face 1 (rows 0-15)
    // Face 2 (rows 16-31) | Face 3 (rows 16-31)
    //
    // Iteration mapping - Process vertically aligned faces (0+2, 1+3) to optimize column operations:
    // i=0: even columns, left half  (faces 0 + 2, columns 0,2,4,6,8,10,12,14)
    // i=1: odd columns,  left half  (faces 0 + 2, columns 1,3,5,7,9,11,13,15)
    // i=2: even columns, right half (faces 1 + 3, columns 16,18,20,22,24,26,28,30)
    // i=3: odd columns,  right half (faces 1 + 3, columns 17,19,21,23,25,27,29,31)
    constexpr uint UPPER_FACE_ADDRS[4] = {0, 0, 16, 16};   // Face 0, 0, 1, 1
    constexpr uint LOWER_FACE_ADDRS[4] = {32, 32, 48, 48}; // Face 2, 2, 3, 3
    constexpr uint COLUMN_OFFSETS[4]   = {0, 2, 0, 2};     // even, odd, even, odd

    // Optimized approach: Process 4 iterations to handle all column combinations
    // This reduces operations by processing complementary face pairs simultaneously, less load/store operations
    for (int i = 0; i < 4; i++)
    {
        // Key optimization: Process faces 0+2 and 1+3 (vertically aligned) instead of 0+1 and 2+3
        // This allows processing all 32 rows of a column at once (16 from upper face + 16 from lower face)
        // Reduces load/store operations by accumulating all rows into one LREG per column group
        // Final result stored in top row of upper face (first row in dest) - no intermediate storage needed

        const uint UPPER_FACE_ADDR = UPPER_FACE_ADDRS[i];
        const uint LOWER_FACE_ADDR = LOWER_FACE_ADDRS[i];
        const uint COLUMN_OFFSET   = COLUMN_OFFSETS[i];

        // Load upper face data (Face 0 or Face 1)
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET);      // rows 0-3
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 4);  // rows 4-7
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 8);  // rows 8-11
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 12); // rows 12-15

        // Load lower face data (Face 2 or Face 3)
        TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET);      // rows 0-3
        TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 4);  // rows 4-7
        TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 8);  // rows 8-11
        TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 12); // rows 12-15

        // Process column sums for both faces using transpose and replay buffer
        TT_SFPTRANSP(0, 0, 0, 0); // Transpose: LREG0-3 → lanes 0-3, LREG4-7 → lanes 0-3 (overlapping)
        lltt::replay(0, 6);       // Column-wise sum within each lreg after transpose
        TT_SFPTRANSP(0, 0, 0, 0); // Transpose back to original register layout
        lltt::replay(0, 6);       // Sum column sums within each face after transpose

        TT_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4); // LREG0 = upper_face_sums + lower_face_sums (integer)

        if constexpr (pool_type == AVG)
        {
            // For integer formats, we need to handle negative numbers properly for division by 32
            // Since Wormhole B0 only supports logical shift (not arithmetic), we need to:
            // 1. Check if the number is negative using condition codes (only for signed formats)
            // 2. If negative, negate it, shift right by 5 bits, then negate back
            // 3. If positive, just shift right by 5 bits

            if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32)
            {
                // For signed Int32 format, use absolute value approach for proper division by 32
                // Save original value for sign check
                TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);

                // Get absolute value of LREG0
                TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

                // Perform logical right shift by 5 bits (divide by 32)
                TTI_SFPSHFT(-5 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);

                // Restore sign if original value was negative
                // Check if original value was negative (sign bit set)
                TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);               // Set condition code if original sign bit is 0 (positive)
                TTI_SFPCOMPC(0, 0, 0, 0);                           // Invert condition code (now true if original was negative)
                TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6); // Negate LREG0 if condition is true

                // Clear condition codes
                TTI_SFPENCC(0, 0, 0, 0);
            }
            else
            {
                // For unsigned formats (UInt16, UInt32), just use logical shift directly
                // since they can't be negative
                TTI_SFPSHFT(-5 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
            }
        }

        // Store the final combined column sums
        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET);
    }

    // After this loop, the column sums are stored at first row in dest reg:
    // Address 0:  even columns, left half  (columns 0,2,4,6,8,10,12,14)
    // Address 2:  odd columns,  left half  (columns 1,3,5,7,9,11,13,15)
    // Address 16: even columns, right half (columns 16,18,20,22,24,26,28,30)
    // Address 18: odd columns,  right half (columns 17,19,21,23,25,27,29,31)
}

inline void init_reduce_int()
{
    // Initialize SFPU configuration register
    _init_sfpu_config_reg();

    // Program optimized replay buffer for column summation
    // This replay buffer is called twice per iteration:
    // 1st call: After first transpose - operates on transposed data where LREG0-3 and LREG4-7 both map to lanes 0→3
    // 2nd call: After second transpose - operates on data transposed back to original layout, the sum of 4 rows columns stored in lregs, need to sum lregs for
    // each face to get the final column sums

    // Program replay buffer using Wormhole lltt::record API
    lltt::record(0, 6);

    // Column summation for upper face data (originally LREG0-3)
    // After transpose: LREG0→lane0, LREG1→lane1, LREG2→lane2, LREG3→lane3 across lregs 0-3
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4); // LREG2 = LREG2 + LREG3
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4); // LREG1 = LREG1 + LREG2
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // LREG0 = LREG0 + LREG1 (upper face column sums)

    // Column summation for lower face data (originally LREG4-7)
    // After transpose: LREG4→lane0, LREG5→lane1, LREG6→lane2, LREG7→lane3 across lregs 4-7
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4); // LREG6 = LREG6 + LREG7
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4); // LREG5 = LREG5 + LREG6
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4); // LREG4 = LREG4 + LREG5 (lower face column sums)

    // The transpose operation allows both upper and lower face data to be processed
    // simultaneously in the same lane space, then separated back to their original registers
}

/**
 * @brief Generic reduction operation for a 32x32 tile, placing output values into the first row.
 *        Currently able to calculate column-wise sum and/or average of a 32x32 tile, placing output values into the first row.
 *        Uses an optimized approach that processes vertically aligned face pairs (0+2, 1+3) to minimize
 *        load/store operations and eliminate intermediate storage requirements.
 *        For floating-point operations, uses native SFPU floating-point arithmetic.
 * @tparam pool_type The pool/reduction pool_type (SUM, AVG, MAX). Currently only SUM and AVG are supported.
 * @tparam reduce_dim The reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR). Currently only REDUCE_COL is supported.
 * @tparam INSTRUCTION_MODE The instruction modifier that determines the data type and precision.
 *                          For float operations, must be InstrModLoadStore::FP32 for 32-bit floating-point.
 */
template <PoolType pool_type, ReduceDim reduce_dim, InstrModLoadStore INSTRUCTION_MODE>
inline void calculate_reduce_float()
{
    // Compile-time assertions to restrict to currently supported operations
    static_assert(reduce_dim == REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported on SFPU Reduce Kernel");
    static_assert(pool_type == SUM || pool_type == AVG, "Only SUM and AVG pool types are currently supported on SFPU Reduce Kernel");
    static_assert(INSTRUCTION_MODE == InstrModLoadStore::FP32, "Only FP32 instruction mode is currently supported on float SFPU Reduce Kernel");

    // Pre-calculated face addresses and column offsets for each iteration
    // Each face is 16 rows, tile has 4 faces arranged as:
    // Face 0 (rows 0-15)  | Face 1 (rows 0-15)
    // Face 2 (rows 16-31) | Face 3 (rows 16-31)
    //
    // Iteration mapping - Process vertically aligned faces (0+2, 1+3) to optimize column operations:
    // i=0: even columns, left half  (faces 0 + 2, columns 0,2,4,6,8,10,12,14)
    // i=1: odd columns,  left half  (faces 0 + 2, columns 1,3,5,7,9,11,13,15)
    // i=2: even columns, right half (faces 1 + 3, columns 16,18,20,22,24,26,28,30)
    // i=3: odd columns,  right half (faces 1 + 3, columns 17,19,21,23,25,27,29,31)
    constexpr uint UPPER_FACE_ADDRS[4] = {0, 0, 16, 16};   // Face 0, 0, 1, 1
    constexpr uint LOWER_FACE_ADDRS[4] = {32, 32, 48, 48}; // Face 2, 2, 3, 3
    constexpr uint COLUMN_OFFSETS[4]   = {0, 2, 0, 2};     // even, odd, even, odd

    // Optimized approach: Process 4 iterations to handle all column combinations
    // This reduces operations by processing complementary face pairs simultaneously, less load/store operations
    for (int i = 0; i < 4; i++)
    {
        // Key optimization: Process faces 0+2 and 1+3 (vertically aligned) instead of 0+1 and 2+3
        // This allows processing all 32 rows of a column at once (16 from upper face + 16 from lower face)
        // Reduces load/store operations by accumulating all rows into one LREG per column group
        // Final result stored in top row of upper face (first row in dest) - no intermediate storage needed

        const uint UPPER_FACE_ADDR = UPPER_FACE_ADDRS[i];
        const uint LOWER_FACE_ADDR = LOWER_FACE_ADDRS[i];
        const uint COLUMN_OFFSET   = COLUMN_OFFSETS[i];

        // Load upper face data (Face 0 or Face 1)
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET);      // rows 0-3
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 4);  // rows 4-7
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 8);  // rows 8-11
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET + 12); // rows 12-15

        // Load lower face data (Face 2 or Face 3)
        TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET);      // rows 0-3
        TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 4);  // rows 4-7
        TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 8);  // rows 8-11
        TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, LOWER_FACE_ADDR + COLUMN_OFFSET + 12); // rows 12-15

        // Process column sums for both faces using transpose and replay buffer
        TT_SFPTRANSP(0, 0, 0, 0); // Transpose: LREG0-3 → lanes 0-3, LREG4-7 → lanes 0-3 (overlapping)
        lltt::replay(0, 12);      // Column-wise sum within each lreg after transpose
        TT_SFPTRANSP(0, 0, 0, 0); // Transpose back to original register layout
        lltt::replay(0, 12);      // Sum column sums within each face after transpose

        // Combine the column sums from upper and lower faces
        TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0); // LREG0 = (LREG0 * 1) + LREG4 = upper_face_sums + lower_face_sums (float)
        TTI_SFPNOP;                                                                  // Required for Wormhole

        if constexpr (pool_type == AVG)
        {
            // For a 32x32 tile, each column sum represents the sum of exactly 32 values (one per row)
            // Load 1/32 constant (0.03125) into LREG1 for float division
            TT_SFPLOADI(p_sfpu::LREG1, 8, 0x3D00);  // Load 0.03125 as FP16B high part
            TT_SFPLOADI(p_sfpu::LREG1, 10, 0x0000); // Load 0.03125 as FP16B low part
            // Multiply by 1/32 (divide by 32) - works for both float and integer formats
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TTI_NOP; // Required after SFPMUL due to 2-cycle latency
        }
        // Store the final combined column sums
        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, UPPER_FACE_ADDR + COLUMN_OFFSET);
    }

    // After this loop, the column sums are stored at first row in dest reg:
    // Address 0:  even columns, left half  (columns 0,2,4,6,8,10,12,14)
    // Address 2:  odd columns,  left half  (columns 1,3,5,7,9,11,13,15)
    // Address 16: even columns, right half (columns 16,18,20,22,24,26,28,30)
    // Address 18: odd columns,  right half (columns 17,19,21,23,25,27,29,31)
}

inline void init_reduce_float()
{
    // Initialize SFPU configuration register
    _init_sfpu_config_reg();

    // Program optimized replay buffer for column summation
    // This replay buffer is called twice per iteration:
    // 1st call: After first transpose - operates on transposed data where LREG0-3 and LREG4-7 both map to lanes 0→3
    // 2nd call: After second transpose - operates on data transposed back to original layout, the sum of 4 rows columns stored in lregs, need to sum lregs for
    // each face to get the final column sums

    // Program replay buffer using Wormhole lltt::record API with NOPs
    lltt::record(0, 12);

    // Column summation for upper face data (originally LREG0-3)
    // After transpose: LREG0→lane0, LREG1→lane1, LREG2→lane2, LREG3→lane3 across lregs 0-3
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0); // LREG2 = (LREG2 * 1) + LREG3 = LREG2 + LREG3 (float)
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0); // LREG1 = (LREG1 * 1) + LREG2 = LREG1 + LREG2 (float)
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0,
               0); // LREG0 = (LREG0 * 1) + LREG1 = LREG0 + LREG1 (upper face column sums, float)
    TTI_SFPNOP;

    // Column summation for lower face data (originally LREG4-7)
    // After transpose: LREG4→lane0, LREG5→lane1, LREG6→lane2, LREG7→lane3 across lregs 4-7
    TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0); // LREG6 = (LREG6 * 1) + LREG7 = LREG6 + LREG7 (float)
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0); // LREG5 = (LREG5 * 1) + LREG6 = LREG5 + LREG6 (float)
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4,
               0); // LREG4 = (LREG4 * 1) + LREG5 = LREG4 + LREG5 (lower face column sums, float)
    TTI_SFPNOP;
}

/**
 * @brief Runtime validation helper for supported data formats for reduce sfpu kernel
 */
constexpr bool is_supported_reduce_format(DataFormat format)
{
    return format == DataFormat::Int32 || format == DataFormat::UInt16 || format == DataFormat::UInt32 || format == DataFormat::Float32;
}

/**
 * @brief Unified reduction operation wrapper for a 32x32 tile, placing output values into the first row.
 *        Automatically chooses between integer and floating-point implementations based on the data format.
 *        Currently able to calculate column-wise sum and/or average of a 32x32 tile, placing output values into the first row.
 *        Uses an optimized approach that processes vertically aligned face pairs (0+2, 1+3) to minimize
 *        load/store operations and eliminate intermediate storage requirements.
 * @tparam pool_type The pool/reduction pool_type (SUM, AVG, MAX). Currently only SUM and AVG are supported.
 * @tparam reduce_dim The reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR). Currently only REDUCE_COL is supported.
 * @tparam format The data format (DataFormat enum value) that determines which implementation to use:
 *                - DataFormat::Int32, UInt16, UInt32: Use integer implementation
 *                - DataFormat::Float32: Uses floating-point initialization for 32-bit floating-point used in sfpu
 */
template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void _calculate_reduce_()
{
    static_assert(reduce_dim == REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(pool_type == SUM || pool_type == AVG, "Only SUM and AVG pool types are currently supported");
    static_assert(is_supported_reduce_format(format), "Unsupported data format. Supported formats: Int32, UInt16, UInt32, Float32");

    if constexpr (format == DataFormat::Int32)
    {
        calculate_reduce_int<pool_type, reduce_dim, InstrModLoadStore::INT32>();
    }
    else if constexpr (format == DataFormat::UInt16)
    {
        calculate_reduce_int<pool_type, reduce_dim, InstrModLoadStore::LO16>();
    }
    else if constexpr (format == DataFormat::UInt32)
    {
        calculate_reduce_int<pool_type, reduce_dim, InstrModLoadStore::INT32_2S_COMP>();
    }
    else if constexpr (format == DataFormat::Float32)
    {
        calculate_reduce_float<pool_type, reduce_dim, InstrModLoadStore::FP32>();
    }
}

/**
 * @brief Unified initialization wrapper for SFPU reduce kernel.
 *        Automatically chooses between integer and floating-point initialization based on the data format.
 * @tparam format The data format (DataFormat enum value) that determines which initialization to use:
 *                - Supported integer formats: Int32, UInt16, UInt32 (uses integer initialization)
 *                - Supported floating-point formats: Float32 (uses floating-point initialization)
 */
template <DataFormat format>
inline void _init_reduce_()
{
    static_assert(is_supported_reduce_format(format), "Unsupported data format. Supported formats: Int32, UInt16, UInt32, Float32");

    if constexpr (format == DataFormat::Int32 || format == DataFormat::UInt16 || format == DataFormat::UInt32)
    {
        // Use integer initialization for integer formats
        init_reduce_int();
    }
    else if constexpr (format == DataFormat::Float32)
    {
        // Use floating-point initialization for Float32 format
        init_reduce_float();
    }
}

} // namespace sfpu
} // namespace ckernel
