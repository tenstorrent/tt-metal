// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// Constants for 32x32 Tile Layout
// ============================================================================
// Each face is 16 rows, tile has 4 faces arranged as:
// Face 0 (rows 0-15)  | Face 1 (rows 0-15)
// Face 2 (rows 16-31) | Face 3 (rows 16-31)

constexpr std::uint32_t NUM_FACES = 4;
constexpr std::uint32_t ROWS_PER_LOAD = 4;

// Constants for averaging (division by 32)
constexpr std::uint32_t AVG_SHIFT_AMOUNT = 5;    // 2^5 = 32
constexpr std::uint32_t AVG_SHIFT_MASK = 0xfff;  // Mask for shift instruction encoding

// Constants for MAX reduction
constexpr std::uint32_t ROWS_PER_TILE = 64;
constexpr std::uint32_t ROWS_PER_FACE = 16;

// Register holding the 0x0000FFFF mask used to clear garbage high bits when loading UInt16 data
// from a 32-bit (fp32 dest accumulation) dest word. Maps to sfpi::vConstIntPrgm0 on Blackhole.
constexpr std::uint32_t CLEAR_REG = p_sfpu::LREG12;

template <bool clear_high_bits>
inline void load_and_clear_high_bits(
    const std::uint32_t lreg_ind,
    const InstrModLoadStore instr_mod0,
    const std::uint32_t sfpu_addr_mode,
    const std::uint32_t dest_reg_addr) {
    TT_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr);
    if constexpr (clear_high_bits) {
        TT_SFPAND(0, CLEAR_REG, lreg_ind, 0);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// Int32 SUM/AVG keep their data in dest as sign-magnitude (matching the MAX/MIN path and the packer),
// but SFPIADD is a two's-complement adder. Wormhole bridges this with the INT32_2S_COMP load/store mode,
// which converts sign-magnitude<->two's-complement in hardware; on Blackhole that conversion is a no-op
// (tenstorrent/tt-llk-bh#16), so we must do it explicitly: convert each operand to two's-complement right
// after load, and convert results back to sign-magnitude right before store.
//
// We use the same SFPCAST+SFPSETSGN primitive as the proven element-wise int kernels (see _add_int_): it is
// direction-neutral (the same cast mode converts both ways) and, crucially, uses no condition codes, so it is
// robust regardless of SFPABS/cc interaction. It requires one free GPR (LREG0-7) as scratch; LREG8-15 are
// constant/config registers that SFPCAST cannot write.
// "Representation swap": converts an integer between its sign-magnitude and two's-complement representations.
constexpr InstrModCast REDUCE_INT_REPRESENTATION_SWAP_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

// Convert one int operand between sign-magnitude and two's-complement, in place. `scratch_gpr` must be a free
// GPR (LREG0-7) distinct from `reg`; it is clobbered. The result is left in `reg`.
inline void convert_int_representation_inplace(std::uint32_t reg, std::uint32_t scratch_gpr) {
    apply_sign_magnitude_conversion(reg, scratch_gpr, REDUCE_INT_REPRESENTATION_SWAP_CAST);
}

/**
 * @brief Load data from face into LREG0-3
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param face_addr Base address of face
 * @param column_offset Column offset for the current iteration, load all rows for even columns (0) or odd columns (2)
 * of the face
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, std::uint32_t DST_LREG_BASE = p_sfpu::LREG0>
inline void load_face_data(std::uint32_t face_addr, std::uint32_t column_offset) {
    // Load the 4 row-groups into DST_LREG_BASE..DST_LREG_BASE+3. DST_LREG_BASE defaults to LREG0, but
    // callers can target LREG4 to feed a recorded swap buffer that operates on LREG4-7 directly, avoiding
    // a redundant LREG0-3 -> LREG4-7 shuffle.
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 0, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 1, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 2, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 3, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Load data from upper and lower faces into LREG0-7
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param upper_face_addr Base address of upper face (Face 0 or Face 1)
 * @param lower_face_addr Base address of lower face (Face 2 or Face 3)
 * @param column_offset Column offset for the current iteration
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void load_face_data(std::uint32_t upper_face_addr, std::uint32_t lower_face_addr, std::uint32_t column_offset) {
    // Load upper face data (Face 0 or Face 1) into LREG0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG3,
        INSTRUCTION_MODE,
        ADDR_MOD_7,
        upper_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15

    // Load lower face data (Face 2 or Face 3) into LREG4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG7,
        INSTRUCTION_MODE,
        ADDR_MOD_7,
        lower_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
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
inline void perform_int_average() {
    if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32) {
        // For signed Int32 format, use absolute value approach for proper division by 32
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // Save original value for sign check
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // Get absolute value of LREG0
        TTI_SFPSHFT(
            -AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            0b01);  // Perform logical right shift by 5 bits (divide by 32)

        // Restore sign if original value was negative
        // Check if original value was negative (sign bit set)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);  // Set condition code if original sign bit is 0 (positive)
        TTI_SFPCOMPC(0, 0, 0, 0);              // Invert condition code (now true if original was negative)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // Negate LREG0 if condition is true
        TTI_SFPENCC(0, 0, 0, 0);                             // Clear condition codes
    } else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
        // Two's-complement signed divide-by-32 (round toward zero). SFPABS clears the sign bit and is
        // only correct for sign-magnitude, so for 2's-complement we take the magnitude via a conditional
        // negate (0 - x), logical-shift, then restore the sign with a second conditional negate.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);      // Save original (2's-complement) value for sign check
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 4);                // cc if sign bit == 0 (non-negative)
        TTI_SFPCOMPC(0, 0, 0, 0);                            // Invert -> cc if negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // LREG0 = -LREG0 (magnitude) when negative
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);  // |x| >> 5 (divide by 32)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);                // cc if original sign bit == 0 (non-negative)
        TTI_SFPCOMPC(0, 0, 0, 0);                            // Invert -> cc if original was negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // Restore sign (2's-complement negate) when negative
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        // For unsigned formats (UInt32), just use logical shift directly since they can't be negative
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    }
}

// Programmable float constant register holding 1/32 (0.03125) for float AVG. Preloaded once by
// init_reduce_sum_avg (only when pool_type == AVG and the format is float), so perform_float_average
// avoids rebuilding the constant via two SFPLOADI on every column group. Maps to LREG12 on Blackhole
// (sfpi::vConstFloatPrgm0); the float reduce path does not use the UInt16 high-bit mask, so this
// register is free to hold the constant across the whole reduce.
constexpr std::uint32_t AVG_RECIP_REG = p_sfpu::LREG12;

/**
 * @brief Perform floating-point averaging (multiply by 1/32)
 *
 * For a 32x32 tile, each column sum represents the sum of exactly 32 values (one per row).
 * This function divides by 32 by multiplying by the constant 1/32 (0.03125), which
 * init_reduce_sum_avg() preloaded into AVG_RECIP_REG (sfpi::vConstFloatPrgm0).
 */
inline void perform_float_average() {
    // Multiply by 1/32 (divide by 32) using the preloaded constant register.
    TTI_SFPMUL(p_sfpu::LREG0, AVG_RECIP_REG, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
}

template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_col_sum_avg() {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
    constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
    constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

    // Optimized column reduction: Reduce → Add → Transpose → HalfReduce
    // Instead of the naive Transpose → Reduce → Transpose → Reduce → Add approach, we first reduce
    // across registers, then add upper+lower faces (all 4 positions carry meaningful partial sums),
    // then transpose, then do a final half-reduce on LREG0-3 only. This eliminates one transpose
    // and halves the second reduction pass, saving 4 instructions per iteration.
    for (std::uint32_t i = 0; i < NUM_FACES; i++) {
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset = COLUMN_OFFSETS[i];

        // Step 1: Tree-reduce across registers (LREG0-3→LREG0, LREG4-7→LREG4) without transpose.
        // After this, each of the 4 positions in LREG0 holds the sum of rows at that position
        // across all 4 loaded LREGs (e.g., LREG0[i] = sum of row[i], row[i+4], row[i+8], row[i+12]).
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
            // Int32: dest holds sign-magnitude but SFPIADD is a two's-complement adder, and converting in place
            // needs a free GPR scratch. With all 8 operands live no scratch is free, so we process the upper and
            // lower faces in sequence: convert+reduce the upper face into LREG0 (freeing LREG1-3), then use those
            // freed registers as scratch to convert+reduce the lower face into LREG4. This reproduces the
            // LREG0=sum(LREG0-3), LREG4=sum(LREG4-7) result of replay(0,6).
            const std::uint32_t upper_base = upper_face_addr + column_offset;
            const std::uint32_t lower_base = lower_face_addr + column_offset;

            // Upper face -> LREG0-3, scratch from the still-free LREG4-7
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, upper_base);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, upper_base + ROWS_PER_LOAD);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, upper_base + 2 * ROWS_PER_LOAD);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, upper_base + 3 * ROWS_PER_LOAD);
            convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
            convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
            convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
            convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);
            TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 += LREG3
            TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 += LREG2
            TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = sum of upper face; LREG1-3 now free

            // Lower face -> LREG4-7, scratch from the now-free LREG1-3
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, lower_base);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, lower_base + ROWS_PER_LOAD);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, lower_base + 2 * ROWS_PER_LOAD);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, lower_base + 3 * ROWS_PER_LOAD);
            convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG1);
            convert_int_representation_inplace(p_sfpu::LREG5, p_sfpu::LREG2);
            convert_int_representation_inplace(p_sfpu::LREG6, p_sfpu::LREG3);
            convert_int_representation_inplace(p_sfpu::LREG7, p_sfpu::LREG1);
            TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 += LREG7
            TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 += LREG6
            TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = sum of lower face
        } else {
            load_face_data<INSTRUCTION_MODE, clear_high_bits>(upper_face_addr, lower_face_addr, column_offset);
            lltt::replay(0, 6);
        }

        // Step 2: Cross-face addition. Unlike the old approach where only position 0 of the
        // cross-face sum was meaningful, here ALL 4 positions carry useful partial sums.
        if constexpr (is_integer_mode) {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
        } else {
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
        }

        // Result of column reduction now stored in LREG0 as 4 partial sums
        // Step 3: Transpose to rearrange the 4 partial sums for final reduction
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed).
        // This sums the 4 partial sums into LREG0[0] = total column sum.
        lltt::replay(6, 3);

        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == PoolType::AVG) {
            if constexpr (is_integer_mode) {
                perform_int_average<INSTRUCTION_MODE>();
            } else {
                perform_float_average();
            }
        }
        // Store the final column sum/average to the first row.
        // Mode 9 (SFPSTORE_MOD0_FMT_LO16) is only needed when the packer-visible OUTPUT is UInt16 in a
        // 32-bit dest: there the reduced value sits in the low 16 bits but the packer reads the high 16,
        // so we move low->high. When the output is a full 32-bit format (e.g. UInt32) the packer reads
        // the whole dest word, so we use the plain INSTRUCTION_MODE store even for UInt16 input.
        // Int32: convert the two's-complement result back to sign-magnitude before storing to dest.
        // LREG1-7 are free here (only LREG0 holds the result), so LREG1 is a safe GPR scratch.
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
            convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG1);
        }
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        // Use the runtime-address store (TT_ not TTI_): the Int32 two's-complement path makes this loop body
        // large enough that GCC no longer unrolls it, so the face address is not a compile-time constant.
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, upper_face_addr + column_offset);
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
inline void horizontal_reduce() {
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

    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(
            p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(
            p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }

    // Phase 2: Shift by 2 and add -> 4 sums become 2 sums (cols 6-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }

    // Phase 3: Shift by 1 and add -> 2 sums become 1 sum (col 7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(
            p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(
            p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }

    // Phase 4: Rotate right by 1 -> move single sum from col 7 to col 0 for store.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
    // LREG0[0 column slice] = sum of all elements in the first 4 rows of this 8-row block (first half)
    // LREG4[0 column slice] = sum of all elements in the next 4 rows of this 8-row block (second half)
}

constexpr std::uint32_t HORIZONTAL_REDUCE_MAX_REPLAY_LEN = 16;

/**
 * @brief Records phases 2-4 of horizontal max reduction into replay buffer at slot 0 (16 instructions).
 *
 * The full horizontal max reduction has 28 instructions across 4 phases:
 *   Phase 1: 2 MOV + 8 SHFT2 + 2 SWAP = 12 (inline, shift-by-4)
 *   Phase 2: 2 MOV + 4 SHFT2 + 2 SWAP =  8 (replay, shift-by-2)
 *   Phase 3: 2 MOV + 2 SHFT2 + 2 SWAP =  6 (replay, shift-by-1)
 *   Phase 4: 2 SHFT2                   =  2 (replay, rotate to col 0)
 *
 * Phase 1 (12 instr) stays inline; phases 2+3+4 (16 instr) fit exactly in one replay buffer.
 * Must be called once before perform_reduce_row_max_tile.
 */
inline void record_horizontal_reduce_max() {
    lltt::record(0, HORIZONTAL_REDUCE_MAX_REPLAY_LEN);

    // Phase 2: Shift by 2 and max -> 4 maxes become 2 maxes (cols 6-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phase 3: Shift by 1 and max -> 2 maxes become 1 max (col 7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phase 4: Rotate right by 1 -> move single max from col 7 to col 0 for store.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
}

/**
 * @brief Executes horizontal max reduction: phase 1 inline, phases 2-4 via replay buffer.
 *        record_horizontal_reduce_max() must have been called before first use.
 */
inline void horizontal_reduce_max() {
    // Phase 1 (inline): Shift by 4 and max -> 8 values become 4 maxes (cols 4-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);

    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phases 2, 3, 4 via replay buffer
    lltt::replay(0, HORIZONTAL_REDUCE_MAX_REPLAY_LEN);
}

/**
 * @brief Row-wise maximum reduction for a single 32x32 tile.
 *
 * Processes the tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31.
 * Each face-pair iteration processes 8 rows (two groups of 4 rows each).
 *
 * For each 8-row group:
 * 1. Load 4 rows from left face (even cols) and 4 rows from right face (odd cols) into LREG0-3
 * 2. Load the next 4 rows into LREG4-7
 * 3. Use vertical SFPSWAP to reduce LREG pairs down (keeping max between left/right face columns)
 * 4. Use horizontal_reduce_max to consolidate 8 SFPU columns into column 0
 * 5. Store the per-row max into column 0
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param tile_row_offset Base row offset for this tile in the dest register
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void perform_reduce_row_max_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
#pragma GCC unroll 2
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

#pragma GCC unroll 2
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Vertical max: reduce left/right face pairs via compare-and-swap.
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

            horizontal_reduce_max();

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is intermediate and stays
            // in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Row-wise maximum reduction for a single 32x32 tile using Int32 values on Blackhole.
 *
 * Due to a Blackhole RTL bug, INT32_2S_COMP load/store has no effect. Data is stored in
 * sign-magnitude format in dest. SFPSWAP(ALL_ROWS_MAX) compares values using the float
 * comparator; sign-magnitude integers have the same ordering as IEEE floats for MAX, so
 * no conversion to 2's complement is needed — we operate directly on sign-magnitude data.
 *
 * @param tile_row_offset Base row offset for this tile in the dest register
 */
template <bool clear_high_bits = false>
inline void perform_reduce_row_max_int32_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;

    perform_reduce_row_max_tile<INSTRUCTION_MODE, clear_high_bits>(tile_row_offset, result_store_mode);
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0.
 *
 * Mirrors sum_first_columns_across_tiles but uses SFPSWAP instead of SFPADD.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void max_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, 1);
        }

        // Final, packer-visible store: use mode 9 (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is
        // UInt16 in a 32-bit dest (pack_low16); a 32-bit output keeps the plain INSTRUCTION_MODE store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0 (Int32).
 *
 * Sign-magnitude integers have the same ordering as IEEE floats for MAX comparison,
 * so we reuse the FP32 variant directly — no 2's complement conversion needed.
 *
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <bool clear_high_bits = false, bool pack_low16 = false>
inline void max_first_columns_across_tiles_int32(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    max_first_columns_across_tiles<InstrModLoadStore::INT32, clear_high_bits, pack_low16>(tile_row_base, block_ct_dim);
}

/**
 * @brief Row-wise maximum reduction across a block of tiles.
 *
 * For each row of tiles, reduces every tile individually, then (if block_ct_dim > 1)
 * accumulates the per-tile column-0 maxima across tiles using compare-and-swap into
 * tile 0's column 0.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 * @param block_rt_dim Number of tiles along y axis of tensor (row tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_max(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    constexpr bool is_int32 = (INSTRUCTION_MODE == InstrModLoadStore::INT32);

    // Row-reduce needs the default MAX SFPSWAP direction (SFPCONFIG bit 8 = 0). That default is already
    // established once by the paired init (init_reduce -> init_reduce_max_min for the MAX pool type,
    // which calls _init_sfpu_config_reg() and does not set bit 8), so we no longer reset the config
    // register on every row-max call here. The only init that flips bit 8 is the column MIN / Int32
    // (sign-magnitude) MAX/MIN path, which is never the init paired with a row-MAX calculate.

    record_horizontal_reduce_max();

    // Single column tile => per-tile store is the final packer-visible result, which uses mode 9 only
    // when the OUTPUT is UInt16 in a 32-bit dest (pack_low16); otherwise it is intermediate and stays
    // in the low 16 bits via INSTRUCTION_MODE.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            if constexpr (is_int32) {
                perform_reduce_row_max_int32_tile(tile_offset, tile_store_mode);
            } else {
                perform_reduce_row_max_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
            }
        }

        if (block_ct_dim > 1) {
            if constexpr (is_int32) {
                max_first_columns_across_tiles_int32<clear_high_bits, pack_low16>(tile_row_offset, block_ct_dim);
            } else {
                max_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                    tile_row_offset, block_ct_dim);
            }
        }
    }
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void perform_reduce_row_sum_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Process tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31
    // Each face-pair iteration processes 8 rows (two groups of 4 rows each)
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        // Base offset for this face pair:
        // face_pair 0: faces 0+1 (dest indices 0-31)
        // face_pair 1: faces 2+3 (dest indices 32-63)
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            // Within each face, process rows in groups of 8 (two sub-groups of 4)
            std::uint32_t row_offset_first = row_group * 8;          // 0 or 8
            std::uint32_t row_offset_second = row_offset_first + 4;  // 4 or 12

            const std::uint32_t group_a_base = tile_row_offset + face_pair_base + row_offset_first;
            const std::uint32_t group_b_base = tile_row_offset + face_pair_base + row_offset_second;

            if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
                // Int32: dest holds sign-magnitude but SFPIADD is a two's-complement adder, and converting in
                // place needs a free GPR scratch. With all 8 operands live no scratch is free, so we process the
                // two independent 4-row groups in sequence: convert+reduce group A into LREG0 (freeing LREG1-3),
                // then use those freed registers as scratch to convert+reduce group B into LREG4. This mirrors
                // the LREG0=sum(LREG0-3), LREG4=sum(LREG4-7) result of replay(0,6).

                // Group A (first 4 rows) -> LREG0-3, scratch from the still-free LREG4-7
                load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE + 2);
                convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
                convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
                convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
                convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);
                TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 += LREG3
                TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 += LREG2
                TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = sum of group A; LREG1-3 now free

                // Group B (next 4 rows) -> LREG4-7, scratch from the now-free LREG1-3
                load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE + 2);
                convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG1);
                convert_int_representation_inplace(p_sfpu::LREG5, p_sfpu::LREG2);
                convert_int_representation_inplace(p_sfpu::LREG6, p_sfpu::LREG3);
                convert_int_representation_inplace(p_sfpu::LREG7, p_sfpu::LREG1);
                TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 += LREG7
                TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 += LREG6
                TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = sum of group B
            } else {
                // Load 4 rows from face 0 (or 2) and face 1 (or 3)
                load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE + 2);

                // Load next 4 rows from face 0 (or 2) and face 1 (or 3)
                load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE + 2);

                // Perform vertical sum of loaded rows via replay buffer
                // After this: LREG0 contains sum of first 4 rows, LREG4 contains sum of next 4 rows
                lltt::replay(0, 6);
            }

            // Horizontal reduction: consolidate all 8 SFPU columns into column 0 (interleaved for latency hiding)
            horizontal_reduce<is_integer_mode>();

            // Int32: convert the two's-complement partial sums back to sign-magnitude before storing.
            // Only LREG0 and LREG4 hold results, so LREG1/LREG5 are free GPR scratch.
            if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
                convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG1);
                convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG5);
            }

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is an intermediate store
            // re-loaded by the cross-tile accumulation and must stay in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Accumulates partial row sums from all tiles in a row of tiles in tensor into tile 0 of that row.
 *
 * After per-tile row reduction, each tile has partial row sums in its column 0.
 * This function accumulates those row sums across all tiles into tile_row_base (first tile in this row of tiles in
 * tensor). Each tile already has 8 partial row-sum results in column 0 (written by perform_reduce_row_sum_tile): 2
 * face-pairs × 2 row-groups × 2 sums per group (first 4 rows and next 4 rows of each 8-row group) = 8. They are stored
 * at row offsets 0, 4, 8, 12, 32, 36, 40, 44. Each result occupies 4 rows (one LREG).
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
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void sum_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Row offset for each of the 8 partial row-sum results (face 0: 0,4,8,12; face 2: 32,36,40,44)
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        // Load tile 0's four LREGs at this batch's offsets (0,4,8,12 or 32,36,40,44) into LREG0-3
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        // Int32: dest holds sign-magnitude; convert tile 0's partial sums to two's-complement so the
        // cross-tile SFPIADD accumulation below is correct (Blackhole INT32_2S_COMP load is a no-op).
        // LREG4-7 are free here, so they serve as GPR scratch for the four casts.
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
            convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
            convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
            convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
            convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);
        }

        // Accumulate from remaining tiles
        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
                // The four accumulators (LREG0-3) are live and converting in place needs a free GPR scratch,
                // so we cannot keep four freshly-loaded operands resident simultaneously. Process one column at
                // a time: load into LREG4, convert to two's-complement (scratch LREG5), then add into its
                // accumulator. LREG5-7 stay free for scratch.
                for (std::uint32_t j = 0; j < 4; j++) {
                    load_and_clear_high_bits<clear_high_bits>(
                        p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + j]);
                    convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG5);
                    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0 + j, 4);  // LREG(j) += LREG4
                }
            } else {
                // Load tile t's four LREGs at the same offsets into LREG4-7
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

                // Add LREG4-7 into LREG0-3
                if constexpr (is_integer_mode) {
                    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
                    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG1, 4);
                    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG2, 4);
                    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 4);
                } else {
                    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
                    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
                    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                    TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
                }
            }
        }

        // Int32: convert the two's-complement accumulated sums back to sign-magnitude before storing.
        // LREG4-7 are free after the accumulation loop, so they serve as GPR scratch for the four casts.
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
            convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
            convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
            convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
            convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);
        }

        // Store LREG0-3 back to tile 0. This is the final, packer-visible result, so it uses mode 9
        // (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is UInt16 in a 32-bit dest (packer reads the
        // high 16 bits); a 32-bit output (e.g. UInt32) is stored with the plain INSTRUCTION_MODE.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_sum(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    // When there is a single column tile, the per-tile store is the final packer-visible result and must
    // use mode 9 only when the OUTPUT is UInt16 in a 32-bit dest (pack_low16). With multiple column tiles
    // the per-tile store is intermediate (re-loaded by sum_first_columns_across_tiles) and must stay in the
    // low 16 bits via INSTRUCTION_MODE; the final mode-9 (if any) is applied by the cross-tile store.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        // Step 1: Reduce each tile individually (horizontal reduction within each tile)
        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_sum_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
        }

        // Step 2: Sum column 0 from all tiles in this row into tile 0's column 0
        if (block_ct_dim > 1) {
            sum_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Runtime validation helper for supported data formats for reduce sfpu kernel
 */
constexpr bool is_supported_reduce_format(DataFormat format) {
    return format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32 ||
           format == DataFormat::Float16_b || format == DataFormat::UInt16;
}

/**
 * @brief Configure address mode for SFPU reduce Max/Min kernel.
 * @param num_cols The number of columns in the tensor block of multiple tiles
 * @note One tile is 64 rows in dest
 */
inline void configure_addrmod_max_min(std::uint32_t num_cols) {
    // Reduction done on first tile before looping through the rest, so we look at num_cols - 1 tile
    std::uint32_t skip_rows = (num_cols - 1) * ROWS_PER_TILE;

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_6);

    addr_mod_t{
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
 * @brief Initialization for SFPU reduce MAX/MIN kernel on 32x32 tile for Int32 format. Due to RTL bug INT32_2S_COMP
 * LOAD/STORE has no effect. Must cast to INT_SIGN_MAGN_TO_INT32_2S_COMP before swapping. Since CAST and SWAP are both
 * SIMPLE instructions, cannot be integrated together in LOADMACRO sequence. Therefore, we need to initialize the kernel
 * with manual loads and stores in order to perform the CAST and SWAP operations.
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min_int32() {
    // Initialize SFPU config and set swap direction
    _init_sfpu_config_reg();
    if constexpr (pool_type == PoolType::MAX) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100);  // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);  // Load upper 16 bits
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
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @param num_cols The number of columns to process (typically 32 for a single tile, or multiple of 32 for block
 * operations)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type, bool clear_high_bits>
inline void init_reduce_max_min(std::uint32_t num_cols) {
    // Initialize SFPU config and set swap direction before defining LOADMACRO sequences
    _init_sfpu_config_reg();

    // Invert swap direction for MIN operations, set 8th bit in SFPU config register
    if constexpr (pool_type == PoolType::MIN) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100);  // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);  // Load upper 16 bits
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

    // Record replay buffer for compare-and-swap operations.
    // Note: this LOADMACRO-based path is only used for float/UInt32 formats. UInt16 in 32-bit dest
    // cannot use it because the fused load+swap leaves no place to mask the garbage high bits, so it
    // is routed to the manual calculate_reduce_max_min_uint16() path instead.
    constexpr std::uint32_t buffer_len = 11;
    lltt::record<lltt::NoExec>(0, buffer_len);
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
 *        Two buffers are recorded:
 *        - Positions 0-5: Full tree reduce for both LREG groups (used by both col and row reduce)
 *        - Positions 6-8: Half tree reduce for LREG0-3 only (used by optimized col reduce)
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_sum_avg() {
    _init_sfpu_config_reg();

    // Determine if integer or float mode based on INSTRUCTION_MODE
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Float AVG divides by 32 by multiplying by 1/32. Preload that constant once here (only when it is
    // actually needed: float AVG) into the programmable float const register AVG_RECIP_REG so
    // perform_float_average() collapses to a single SFPMUL instead of rebuilding the constant with two
    // SFPLOADI on every column group. Integer AVG uses perform_int_average (shift) and pays nothing; SUM
    // never averages, so it pays nothing either.
    if constexpr (pool_type == PoolType::AVG && !is_integer_mode) {
        sfpi::vConstFloatPrgm0 = 0.03125f;
    }

    // Record two replay buffers:
    // Positions 0-5: Full tree reduce (both LREG groups, interleaved for latency hiding)
    //   - Used by column reduce (first pass) and row reduce
    // Positions 6-8: Half tree reduce (LREG0-3 only)
    //   - Used by optimized column reduce (second pass, after cross-face add + transpose)

    if constexpr (is_integer_mode) {
        lltt::record(0, 9);

        // Full reduce (positions 0-5): interleaved upper/lower face summation
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 = LREG6 + LREG7
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 = LREG1 + LREG2
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 = LREG5 + LREG6
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = LREG0 + LREG1
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = LREG4 + LREG5

        // Half reduce (positions 6-8): upper face only (LREG0-3)
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 = LREG1 + LREG2
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = LREG0 + LREG1
    } else {
        lltt::record(0, 9);

        // Full reduce (positions 0-5): interleaved to eliminate read-after-write dependencies
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);  // B1
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // B2
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // B3

        // Half reduce (positions 6-8): upper face only (LREG0-3), no NOPs needed on Blackhole
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
    }
}

// ============================================================================
// Calculate Functions
// ============================================================================

/**
 * @brief Column-wise maximum/minimum reduction kernel for UInt16 stored in a 32-bit (fp32 dest acc) dest.
 *
 * UInt16 datums live in the low 16 bits of a 32-bit dest word and carry garbage in the high 16 bits,
 * so every value must be masked (AND 0x0000FFFF) before it participates in a compare-and-swap. The
 * float/UInt16 LOADMACRO pipeline used by calculate_reduce_max_min() fuses the load and the swap into
 * a single macro, leaving no place to clear the high bits in between; the garbage then dominates the
 * integer comparison and produces wrong minima/maxima for any non-constant input.
 *
 * Instead we use the same manual load/mask/swap structure as the Int32 path (which exists for the
 * analogous reason that CAST and SWAP cannot share a LOADMACRO). The sign-magnitude casts of the
 * Int32 path are unnecessary here: masked UInt16 values are always non-negative, so a plain integer
 * SFPSWAP orders them correctly and the cast would be the identity. We therefore replace the casts
 * with plain moves (LREG0-3 -> LREG4-7) that feed the recorded swap buffer. Final, packer-visible
 * results are written with SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16) so the packer reads the low bits.
 *
 * The reduction reuses init_reduce_max_min_int32()'s 3-swap replay buffer and swap-direction config.
 * Only a single 32x32 tile is processed per call (block height 1), matching the column-reduce driver
 * which invokes the kernel once per tile.
 *
 * @tparam INSTRUCTION_MODE The instruction mode (INT32 for UInt16 in 32-bit dest)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in 32-bit dest)
 * @tparam pack_low16 Whether the final packer-visible store uses mode 9 (true for UInt16 OUTPUT in 32-bit dest)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min_uint16() {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    constexpr std::uint32_t ODD_COLUMNS = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2};  // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4] = {
        {0, 0, 32, 32},   // j=0: Face 0 and Face 2
        {16, 16, 48, 48}  // j=1: Face 1 and Face 3
    };
    constexpr std::uint32_t FINAL_REDUCE_ADDRS[2][2] = {
        {0, 32},  // j=0: Face 0 and Face 2
        {16, 48}  // j=1: Face 1 and Face 3
    };

    // The intermediate stores below land in non-row-0 dest slots that we reload, so they use the
    // plain (full 32-bit) instruction mode. Only the final row-0 stores are packer-visible and use
    // mode 9 (SFPSTORE_MOD0_FMT_LO16) when the OUTPUT is UInt16 in a 32-bit dest (pack_low16), so the
    // packer reads the low 16 bits of the dest word.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = FINAL_REDUCE_ADDRS[j][0];     // face 0 & 1 dst indices
        std::uint32_t bottom_face_addr = FINAL_REDUCE_ADDRS[j][1];  // face 2 & 3 dst indices

        // Reduce each of the four vertically adjacent faces (f0,f2 then f1,f3) within itself; the
        // max/min of its 16 rows is left in the top 4 rows.
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            // Masked load straight into LREG4-7, where the recorded swap buffer reduces them. Loading
            // directly into the target registers eliminates the four LREG0-3 -> LREG4-7 moves (and the
            // post-reduce LREG4 -> LREG0 move) that the previous LREG0-3 load required.
            load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);

            lltt::replay(0, 3);  // compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPSTORE(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, FACE_ADDRS[j][i] + COLUMN_OFFSETS[i]);
        }

        // Load the partial max/min (top 4 rows) of the two vertically adjacent faces into LREG0-3.
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr);
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr + ODD_COLUMNS);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr + ODD_COLUMNS);

        // Move into LREG4-7 for the transpose + cross-row reduction.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

        // Transpose so the 4 partial results of each column sit in one register, reduce, transpose back.
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(0, 3);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Swap to combine the two vertically adjacent faces (even and odd columns).
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);  // odd columns of face pair
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);  // even columns of face pair

        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, top_face_addr + ODD_COLUMNS);
    }
}

/**
 * @brief Column-wise maximum/minimum reduction kernel for SFPU reduce MAX/MIN operation.
 *        Processes a block of tiles vertically (block_height tiles stacked) and computes the maximum or minimum value
 *        for each of the columns across all rows in the block. The maximum/minimum values are placed into
 *        the first row of the output tile (row 0 of faces 0 and 1) in tilized format for each tile in the top row of
 * tiles in the block.
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
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @param block_height The number of tiles in the vertical block to reduce (default is 1 for single tile).
 *                     For example, block_height=4 means reduce across 4 vertically stacked tiles (128 rows total).
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min(const std::uint32_t block_height) {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    // Per-face-pair replay window and the dummy-load tail in the recorded LOADMACRO buffer.
    constexpr std::uint32_t replay_buffer_offset = 9;
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
    for (std::uint32_t i = 0; i < block_height - 1; i++) {
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

    // Store results to first row.
    // For UInt16 OUTPUT in a 32-bit dest the reduced value lives in the low 16 bits of the LREG, but
    // the packer reads the high 16 bits of the dest word. SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16)
    // writes the low 16 bits into the half the packer consumes; a 32-bit output keeps the plain store.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
    TTI_SFPSTORE(p_sfpu::LREG4, STORE_MODE, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, STORE_MODE, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, STORE_MODE, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG7, STORE_MODE, ADDR_MOD_7, 18);
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
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    // Compile-time assertions to restrict to currently supported operations
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == ReduceDim::REDUCE_ROW),
        "Only column reduction (REDUCE_COL) is supported, except row reduction (REDUCE_ROW) is allowed only for SUM");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "Only SUM and AVG pool types are currently supported on SFPU");

    // Supported instruction modes for SFPU reduce sum/avg (integer and float)
    constexpr bool is_supported_reduce_instr_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16 || INSTRUCTION_MODE == InstrModLoadStore::FP32 ||
         INSTRUCTION_MODE == InstrModLoadStore::FP16B);
    static_assert(
        is_supported_reduce_instr_mode, "INSTRUCTION_MODE must be one of: INT32, INT32_2S_COMP, LO16, FP32, FP16B");

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        perform_reduce_col_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
    } else {
        static_assert(pool_type == PoolType::SUM, "Row reduction (REDUCE_ROW) is allowed only for SUM");
        perform_reduce_row_sum<INSTRUCTION_MODE, clear_high_bits, pack_low16>(block_ct_dim, block_rt_dim);
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
 * @param block_ct_dim Block dimension (used for MAX/MIN reduction to specify number of columns, default is 1 for single
 * tile)
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_acc_en>
inline void init_reduce(std::uint32_t block_ct_dim = 1) {
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs. Int32 SUM/AVG use INT32_2S_COMP so SFPIADD operates on
    // two's-complement values (on Blackhole the load/store conversion is a no-op, so the reduce code casts
    // sign-magnitude<->two's-complement explicitly). Int32 MAX/MIN keep plain INT32 (sign-magnitude):
    // SFPSWAP(VEC_MIN_MAX) is a float/sign-magnitude comparator that orders sign-magnitude integers
    // correctly, whereas two's-complement negatives would be mis-ordered.
    constexpr bool int32_sum_avg =
        (format == DataFormat::Int32 && (pool_type == PoolType::SUM || pool_type == PoolType::AVG));
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        int32_sum_avg ? InstrModLoadStore::INT32_2S_COMP : GetSfpLoadStoreInstrMod<format, is_fp32_dest_acc_en>();

    // Garbage high bits needs to be cleared when loading UInt16 data
    constexpr bool clear_high_bits = (is_fp32_dest_acc_en && format == DataFormat::UInt16);

    if constexpr (clear_high_bits) {
        // CLEAR_REG (sfpi::vConstIntPrgm0 / LREG12) holds the mask used by SFPLOAD_EXT to clear high bits.
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }

    // Dispatch to appropriate PoolType init
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest uses the manual (non-LOADMACRO) compare-and-swap path so the
            // garbage high bits can be masked before each swap. It reuses the Int32 path's 3-swap
            // replay buffer and swap-direction config (the body is format-agnostic).
            init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
        } else {
            init_reduce_max_min<INSTRUCTION_MODE, pool_type, false>(block_ct_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        init_reduce_sum_avg<INSTRUCTION_MODE, pool_type>();
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
                pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

/**
 * @brief Unified reduction kernel wrapper for a 32x32 tile.
 *        Determines the instruction mode from format, then dispatches to the appropriate reduction kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam reduce_dim The reduction dimension: REDUCE_COL for column-wise, REDUCE_ROW for row-wise (MAX only,
 * FP32/Int32).
 * @tparam format The INPUT data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b). Drives the
 *         instruction mode and load-time high-bit masking.
 * @tparam output_format The packer-visible OUTPUT data format (defaults to @p format). Drives the final store mode:
 *         UInt16 output in a 32-bit dest is stored via mode 9 (low->high 16-bit swap), while a 32-bit output (e.g.
 *         UInt32) is stored with the plain instruction mode. This lets UInt16 input be summed into a UInt32 output
 *         without overflow.
 * @param block_ct_dim Block dimension (used for SUM/AVG column reduction to specify number of columns, default is 1 for
 * single tile)
 * @param block_rt_dim Block dimension (used for MAX/MIN reduction to specify block height, or SUM/MAX row reduction;
 * default is 1 for single tile)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    bool is_fp32_dest_acc_en,
    DataFormat output_format = format>
inline void calculate_reduce(std::uint32_t block_ct_dim = 1, std::uint32_t block_rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == ReduceDim::REDUCE_ROW) ||
            (pool_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Row reduction (REDUCE_ROW) is supported for SUM and MAX pool types");
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs.
    // Int32 SUM/AVG use INT32_2S_COMP so SFPIADD operates on two's-complement values (on Blackhole the
    // load/store conversion is a no-op, so the reduce code casts sign-magnitude<->two's-complement
    // explicitly). Int32 MAX/MIN (both dims) keep plain INT32 (sign-magnitude): SFPSWAP(VEC_MIN_MAX) is a
    // float/sign-magnitude comparator that orders sign-magnitude integers correctly; two's-complement
    // negatives are mis-ordered (min of negatives would return the least-negative value).
    constexpr bool int32_sum_avg =
        (format == DataFormat::Int32 && (pool_type == PoolType::SUM || pool_type == PoolType::AVG));
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        int32_sum_avg ? InstrModLoadStore::INT32_2S_COMP : GetSfpLoadStoreInstrMod<format, is_fp32_dest_acc_en>();

    // Garbage high bits needs to be cleared when loading UInt16 data (driven by INPUT format).
    constexpr bool clear_high_bits = (is_fp32_dest_acc_en && format == DataFormat::UInt16);

    // The packer-visible result must go through mode-9 (low->high 16-bit) store only when the OUTPUT is UInt16
    // in a 32-bit dest. A 32-bit output (e.g. UInt32) keeps the full word, so it uses the plain store. This is
    // driven by the OUTPUT format and is independent of the load-time masking above.
    constexpr bool pack_low16 = (is_fp32_dest_acc_en && output_format == DataFormat::UInt16);

    // Dispatch to appropriate reduction kernel based on PoolType
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
            static_assert(
                pool_type == PoolType::MAX, "Row reduction (REDUCE_ROW) currently only supports MAX pool type");
            static_assert(
                INSTRUCTION_MODE == InstrModLoadStore::FP32 || INSTRUCTION_MODE == InstrModLoadStore::INT32 ||
                    INSTRUCTION_MODE == InstrModLoadStore::FP16B,
                "Row MAX reduction supports FP32, FP16B, and INT32 (sign-magnitude) instruction modes");
            perform_reduce_row_max<INSTRUCTION_MODE, clear_high_bits, pack_low16>(block_ct_dim, block_rt_dim);
        } else if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest: manual load/mask/swap path (LOADMACRO cannot mask between load and swap).
            calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
        } else {
            calculate_reduce_max_min<pool_type, reduce_dim, INSTRUCTION_MODE, false, pack_low16>(block_rt_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        calculate_reduce_sum_avg<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
            block_ct_dim, block_rt_dim);
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
                pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

}  // namespace sfpu
}  // namespace ckernel
