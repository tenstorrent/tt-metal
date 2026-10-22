// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/numeric/bfloat16.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
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
constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

constexpr std::uint32_t ROWS_PER_LOAD = 4;

// Constants for averaging (division by 32)
constexpr std::uint32_t AVG_SHIFT_AMOUNT = 5;    // 2^5 = 32
constexpr std::uint32_t AVG_SHIFT_MASK = 0xfff;  // Mask for shift instruction encoding

// Constants for MAX/MIN reduction
constexpr std::uint32_t ROWS_PER_TILE = 64;
constexpr std::uint32_t ROWS_PER_FACE = 16;

// Register holding the 0x0000FFFF mask used to clear garbage high bits when loading UInt16 data
// from a 32-bit (fp32 dest accumulation) dest word. Maps to sfpi::vConstIntPrgm0 on Wormhole B0.
constexpr std::uint32_t CLEAR_REG = p_sfpu::LREG12;

// SFPLOAD wrapper that optionally clears the high 16 bits of the loaded value. Needed for UInt16
// reduce with fp32 dest accumulation: the datum lives in the low 16 bits of a 32-bit dest word and
// the high 16 bits are garbage, which would otherwise corrupt integer compare/add operations.
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

// Number of SFPU instructions emitted by _emit_int32_signed_cswap_ (one two's-complement compare-and-swap).
constexpr std::uint32_t INT32_SIGNED_CSWAP_LEN = 5;

/**
 * @brief Two's-complement signed compare-and-swap of a register pair, matching SFPSWAP(VEC_MIN_MAX)
 *        semantics but correct for the full Int32 range (including INT32_MIN).
 *
 * SFPSWAP(VEC_MIN_MAX) compares in sign-magnitude (it is the float/sign-magnitude comparator). For
 * two's-complement operands that is correct except when BOTH operands are negative: there the
 * sign-magnitude order is the exact reverse of the two's-complement order. It is also the case that
 * INT32_MIN (0x80000000) reads as sign-magnitude "negative zero" (magnitude 0, ranked as 0), so a
 * plain sign-magnitude compare drops it. Both problems are cured by loading the operands with plain
 * INT32 (so the two's-complement bits are preserved, not converted to sign-magnitude) and then
 * re-exchanging the pair on lanes where both operands are negative. The re-swap is SFPSWAP_MOD1_SWAP
 * (unconditional exchange) gated by the condition code, so it is unaffected by the MAX/MIN direction
 * config, and because it is a symmetric exchange it leaves the corrected extreme in whichever of the
 * two registers the caller reads next (matching the original bare SFPSWAP). This mirrors the fix used
 * by calculate_binary_max_min_int32 in ckernel_sfpu_binary_max_min.h.
 *
 * @tparam HIGH_LREG The srcC register of the compare-and-swap.
 * @tparam LOW_LREG  The dest register of the compare-and-swap.
 */
template <std::uint32_t HIGH_LREG, std::uint32_t LOW_LREG>
inline void _emit_int32_signed_cswap_() {
    TTI_SFPSWAP(0, HIGH_LREG, LOW_LREG, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSETCC(0, LOW_LREG, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);   // cc: LOW_LREG < 0
    TTI_SFPSETCC(0, HIGH_LREG, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc &= HIGH_LREG < 0 (both negative)
    TTI_SFPSWAP(0, HIGH_LREG, LOW_LREG, sfpi::SFPSWAP_MOD1_SWAP);
    TTI_SFPENCC(0, 0, 0, 0);
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
        p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG3,
        INSTRUCTION_MODE,
        ADDR_MOD_3,
        upper_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15

    // Load lower face data (Face 2 or Face 3) into LREG4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG7,
        INSTRUCTION_MODE,
        ADDR_MOD_3,
        lower_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Load all 16 rows of a single face into LREG0-3 (4 rows per LREG)
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in 32-bit dest)
 * @param face_addr Base address of face
 * @param column_offset Column offset for the current iteration: even columns (0) or odd columns (2) of the face
 * @tparam DST_LREG_BASE First of the four consecutive LREGs to load into (defaults to LREG0).
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, std::uint32_t DST_LREG_BASE = p_sfpu::LREG0>
inline void load_face_data(std::uint32_t face_addr, std::uint32_t column_offset) {
    // Load the 4 row-groups into DST_LREG_BASE..DST_LREG_BASE+3. DST_LREG_BASE defaults to LREG0, but
    // callers can target LREG4 to feed a recorded swap buffer that operates on LREG4-7 directly, avoiding
    // a redundant LREG0-3 -> LREG4-7 shuffle.
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 0, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 1, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 2, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 3, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Integer divide-by-32 for column AVG, rounding toward zero
 * @tparam INSTRUCTION_MODE The load/store mode the column path runs under.
 * @tparam is_signed_int True for signed Int32, whose LREG0 holds a two's-complement sum; false for the
 *         unsigned formats (UInt32, UInt16), whose sum is an unsigned 32-bit value.
 *
 * Wormhole B0 only has a logical right shift, so the signed path takes the magnitude, shifts it and
 * restores the sign, while the unsigned path shifts directly. The average is left in
 * int_average_result_lreg<is_signed_int>(): LREG1 for the signed path, which keeps the original sum in
 * LREG0 for its sign check, and LREG0 otherwise. The arm must follow the data's
 * signedness, not the load mode: Int32 and UInt32 both load with INT32, and treating an unsigned
 * column sum with bit 31 set as negative returns a wrong, negated average (tt-metal#57509).
 */
template <bool is_signed_int>
constexpr std::uint32_t int_average_result_lreg() {
    return is_signed_int ? p_sfpu::LREG1 : p_sfpu::LREG0;
}

template <InstrModLoadStore INSTRUCTION_MODE, bool is_signed_int>
inline void perform_int_average() {
    // Both arms operate on the raw DEST word. INT32_2S_COMP would reinterpret a two's-complement
    // operand as sign-magnitude on load and store, turning every negative average into its magnitude.
    static_assert(
        INSTRUCTION_MODE != InstrModLoadStore::INT32_2S_COMP,
        "Integer AVG operates on raw two's-complement or unsigned words; do not load it with INT32_2S_COMP");

    if constexpr (is_signed_int) {
        // Two's-complement signed divide-by-32 (round toward zero): shift the magnitude, then negate
        // the lanes whose sum was negative. Integer SFPABS is a two's-complement abs (tt-isa SFPABS.md)
        // and leaves INT32_MIN as 0x80000000, which the logical shift still maps to 2^26 and the
        // negate to INT32_MIN / 32. The magnitude is built in LREG1 so LREG0 keeps the signed sum.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPABS_MOD1_INT);                   // LREG1 = |x|
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG1, p_sfpu::LREG1, 0b01);  // |x| >> 5 (divide by 32)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc if the sum was negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6);               // Restore sign (2's-complement negate)
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        // Unsigned formats (UInt32, UInt16): a logical shift is already the exact divide.
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    }
}

// Programmable float constant register holding 1/32 (0.03125) for float AVG. Preloaded once by
// init_reduce_sum_avg (only when pool_type == AVG and the format is float), so the column AVG multiply
// in perform_reduce_col_sum_avg avoids rebuilding the constant via two SFPLOADI on every column group. Maps to LREG12
// on Wormhole B0 (sfpi::vConstFloatPrgm0); the float reduce path does not use the UInt16 high-bit mask, so this
// register is free to hold the constant across the whole reduce.
constexpr std::uint32_t AVG_RECIP_REG = p_sfpu::LREG12;

/**
 * @brief Row-average reciprocal (1 / num_cols) split into the two 16-bit halves an SFPLOADI accepts.
 *
 * A row reduction collapses every column of a row of tiles, so the average divides the row sum by
 * num_cols = 32 * block_ct_dim columns. The column path always divides by the fixed 32 rows of a
 * tile and can use the compile-time 1/32 constant the init preloads into the programmable float
 * const register; the row divisor instead depends on the runtime block_ct_dim. A runtime value
 * cannot be written into that programmable const register with a plain SFPLOADI (it requires a
 * config write), so the row path keeps the reciprocal in an ordinary working LREG and multiplies
 * with it directly (same idiom as the Welford's reciprocal load). This struct carries the precomputed
 * 16-bit halves so each divide site can reload the reciprocal into a scratch register.
 */
struct RowAvgReciprocal {
    std::uint16_t high16 = 0;
    std::uint16_t low16 = 0;
};

inline RowAvgReciprocal make_row_avg_reciprocal(std::uint32_t num_cols) {
    const FloatBits bits(1.0f / static_cast<float>(num_cols));
    return RowAvgReciprocal{bits.high16, bits.low16};
}

/**
 * @brief Load the row-average reciprocal into @p scratch_lreg (an SFPLOADI pair).
 */
inline void load_row_avg_reciprocal_into(std::uint32_t scratch_lreg, RowAvgReciprocal recip) {
    TT_SFPLOADI(scratch_lreg, sfpi::SFPLOADI_MOD0_UPPER, recip.high16);
    TT_SFPLOADI(scratch_lreg, sfpi::SFPLOADI_MOD0_LOWER, recip.low16);
}

template <
    PoolType pool_type,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void perform_reduce_col_sum_avg() {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Optimized column reduction: Reduce → Add → Transpose → HalfReduce
    // Instead of the naive Transpose → Reduce → Transpose → Reduce → Add approach, we first reduce
    // across registers, then add upper+lower faces (all 4 positions carry meaningful partial sums),
    // then transpose, then do a final half-reduce on LREG0-3 only. This eliminates one transpose
    // and halves the second reduction pass.
    // Float arm only: SFPADD / SFPMUL need a cycle before their result is read, which the original body paid
    // as SFPNOPs. SFPTRANSP consumes LREG4-7 too, so loads of the NEXT face pair's lower face cannot precede
    // it, but the two NOP slots after it (after the half reduce, and after the AVG multiply) can each take one
    // such load: the next iteration then issues 1 (SUM) or 2 (AVG) fewer loads. The lower face of face pair 0
    // is loaded here in a prologue; the integer arm has no NOPs to fill and keeps the plain order.
    if constexpr (!is_integer_mode) {
        load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4>(LOWER_FACE_ADDRS[0], COLUMN_OFFSETS[0]);
    }

#pragma GCC unroll 4
    for (std::uint32_t i = 0; i < NUM_FACES; i++) {
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset = COLUMN_OFFSETS[i];
        const bool has_next = (i + 1 < NUM_FACES);
        const std::uint32_t next_lower_base = has_next ? LOWER_FACE_ADDRS[i + 1] + COLUMN_OFFSETS[i + 1] : 0;
        if constexpr (is_integer_mode) {
            load_face_data<INSTRUCTION_MODE, clear_high_bits>(upper_face_addr, lower_face_addr, column_offset);
        } else {
            load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG0>(upper_face_addr, column_offset);
            if (i > 0) {
                // LREG4 (and LREG5 for AVG) of this lower face were loaded in the previous iteration's NOP slots.
                if constexpr (pool_type != PoolType::AVG) {
                    load_and_clear_high_bits<clear_high_bits>(
                        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + ROWS_PER_LOAD);
                }
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + 3 * ROWS_PER_LOAD);
            }
        }

        // Step 1: Tree-reduce across registers (LREG0-3→LREG0, LREG4-7→LREG4) without transpose.
        lltt::replay(0, 6);

        // Step 2: Cross-face addition — all 4 positions carry meaningful partial sums.
        if constexpr (is_integer_mode) {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
        } else {
            TTI_SFPNOP;  // Wait for LREG4 (2-cycle SFPADD latency)
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
            TTI_SFPNOP;                                                             // Wait for LREG0 before SFPTRANSP
        }

        // Step 3: Transpose to rearrange the 4 partial sums for final reduction
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed).
        if constexpr (is_integer_mode) {
            lltt::replay(0, 3);  // Reuse first 3 positions of sequential integer buffer (A1,A2,A3)
        } else {
            lltt::replay(6, 4);  // Half reduce with NOP for 2-cycle SFPADD latency
            // The half reduce ends with an SFPADD producing LREG0; its next consumer (SFPSTORE for SUM, the
            // SFPMUL for AVG) needs one cycle. LREG4-7 are dead after the transpose, so that slot takes the
            // next face pair's first lower load instead of a NOP (a NOP on the last face pair).
            if (has_next) {
                load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, next_lower_base);
            } else {
                TTI_SFPNOP;
            }
        }

        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == PoolType::AVG) {
            if constexpr (is_integer_mode) {
                perform_int_average<INSTRUCTION_MODE, is_signed_int>();
            } else {
                // Multiply by the preloaded 1/32 (AVG_RECIP_REG); the store needs one cycle after it, which takes
                // the next face pair's second lower load (a NOP on the last face pair).
                TTI_SFPMUL(p_sfpu::LREG0, AVG_RECIP_REG, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                if (has_next) {
                    load_and_clear_high_bits<clear_high_bits>(
                        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, next_lower_base + ROWS_PER_LOAD);
                } else {
                    TTI_SFPNOP;
                }
            }
        }
        // Store the final column sum/average to the first row.
        // Mode 9 (SFPSTORE_MOD0_FMT_LO16) is only needed when the packer-visible OUTPUT is UInt16 in a
        // 32-bit dest: there the reduced value sits in the low 16 bits but the packer reads the high 16,
        // so we move low->high. When the output is a full 32-bit format (e.g. UInt32) the packer reads
        // the whole dest word, so we use the plain INSTRUCTION_MODE store even for UInt16 input.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        constexpr std::uint32_t RESULT_LREG =
            (pool_type == PoolType::AVG && is_integer_mode) ? int_average_result_lreg<is_signed_int>() : p_sfpu::LREG0;
        TTI_SFPSTORE(RESULT_LREG, STORE_MODE, ADDR_MOD_3, upper_face_addr + column_offset);
    }
}

/**
 * @brief Rotate LREG1 and LREG5 right by one column, in place (one butterfly step for both pairs).
 */
inline void horizontal_reduce_rotate_step() {
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

/**
 * @brief LREG1 = ror1(LREG0), LREG5 = ror1(LREG4): the first rotate of every butterfly stage. SFPSHFT2
 *        SUBVEC_SHFLROR1 writes VD from VC and VD may differ from VC (tt-isa SFPSHFT2.md), so the stage reads
 *        the accumulator directly instead of copying it with SFPMOV first.
 */
inline void horizontal_reduce_rotate_first() {
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

/**
 * @brief LREG1 = ror<shift>(LREG0) and LREG5 = ror<shift>(LREG4), 2 * shift instructions, no SFPMOV: the first
 *        rotate of each pair reads the accumulator, the rest rotate in place.
 */
template <std::uint32_t shift>
inline void horizontal_reduce_rotate() {
    static_assert(shift == 4 || shift == 2 || shift == 1, "butterfly stages rotate by 4, 2 or 1 columns");
    horizontal_reduce_rotate_first();
    if constexpr (shift >= 2) {
        horizontal_reduce_rotate_step();
    }
    if constexpr (shift >= 4) {
        horizontal_reduce_rotate_step();
        horizontal_reduce_rotate_step();
    }
}

// ============================================================================
// Replay buffer slot map
// ============================================================================
// cmath_common.h splits the per-thread 32-slot replay buffer: [0, ckernel::math::replay_buf_offset) = [0, 16) is the
// SFPU half and [16, 32) belongs to the FPU (matmul, the FPU reduce and transpose_dest record their bodies there at
// init and replay them per tile), so every window below ends at or below replay_buf_offset. All of them are recorded
// by init_reduce; no calculate records. One init serves REDUCE_COL and REDUCE_ROW, so each row window sits above the
// column window of the same init:
//
//   init                                       column window          row window
//   SUM / AVG, float                           tree-add [0, 10)       butterfly stage [10, 16)
//   SUM / AVG, integer                         tree-add [0, 6)        butterfly stage [9, 16)
//   MAX / MIN, float or 16-bit UInt16          LOADMACRO [0, 9)       butterfly stage [10, 16)
//   MAX / MIN, DISABLE_SFPLOADMACRO            manual swaps [0, 3)    butterfly stage [10, 16)
//   MAX / MIN, UInt16 in a 32-bit Dest         manual swaps [0, 3)    none (column only)
//   MAX / MIN, Int32 / UInt32                  signed swaps [0, 15)   none (the row reduce is inline)
//
// A row reduce replays what its init recorded, so it needs a fresh init_reduce after anything else records into the
// SFPU half.
constexpr std::uint32_t COL_SUM_INT_REPLAY_LEN = 6;     // 6 SFPIADD; [0, 3) doubles as the half reduce
constexpr std::uint32_t COL_SUM_FLOAT_REPLAY_LEN = 10;  // 6 SFPADD full reduce + 4-slot half reduce
constexpr std::uint32_t COL_MAX_MANUAL_REPLAY_LEN = 3;  // 3 SFPSWAP: LREG4-7 -> LREG4
constexpr std::uint32_t COL_MAX_INT32_SIGNED_REPLAY_LEN = 3 * INT32_SIGNED_CSWAP_LEN;

// The LOADMACRO column MAX/MIN window: two more SFPAND when masking UInt16 high bits.
template <bool clear_high_bits>
constexpr std::uint32_t col_max_loadmacro_replay_len() {
    return clear_high_bits ? 11 : 9;
}

/**
 * @brief How a butterfly stage folds the rotated copies into the accumulators (LREG0 += LREG1, LREG4 += LREG5, or the
 *        compare-and-swap equivalent).
 */
enum class HorizontalFold { ADD_FLOAT, ADD_INT, MIN_MAX };

/**
 * @brief Fold the rotated copies into the accumulators. Wormhole forbids SFPIADD on the cycle after an SFPSHFT2
 *        SHFLROR1 (tt-isa SFPSHFT2.md scheduling list); SFPADD and SFPSWAP are not on that list, so only the integer
 *        add pays one SFPNOP. MIN_MAX keeps the extreme in LREG0 / LREG4 (direction = SFPCONFIG bit 8 at issue time).
 */
template <HorizontalFold fold>
inline void horizontal_reduce_fold() {
    if constexpr (fold == HorizontalFold::ADD_INT) {
        TTI_SFPNOP;
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else if constexpr (fold == HorizontalFold::ADD_FLOAT) {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    } else {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
    }
}

// One SFPSHFT2 per accumulator pair.
constexpr std::uint32_t HORIZONTAL_ROTATE_LEN = 2;

template <HorizontalFold fold>
constexpr std::uint32_t horizontal_fold_len() {
    return fold == HorizontalFold::ADD_INT ? 3 : 2;
}

/**
 * @brief The row butterfly-stage window, [A, s, F] = first rotate, one in-place rotate, fold: 6 slots, 7 for the
 *        integer add. It ends at replay_buf_offset, directly above the column window of the same init.
 */
template <HorizontalFold fold>
constexpr std::uint32_t horizontal_stage_replay_len() {
    return 2 * HORIZONTAL_ROTATE_LEN + horizontal_fold_len<fold>();
}
template <HorizontalFold fold>
constexpr std::uint32_t horizontal_stage_replay_start() {
    return ckernel::math::replay_buf_offset - horizontal_stage_replay_len<fold>();
}

static_assert(
    COL_SUM_FLOAT_REPLAY_LEN <= horizontal_stage_replay_start<HorizontalFold::ADD_FLOAT>(),
    "float row SUM/AVG window overlaps the column tree-add window");
static_assert(
    COL_SUM_INT_REPLAY_LEN <= horizontal_stage_replay_start<HorizontalFold::ADD_INT>(),
    "integer row SUM window overlaps the column tree-add window");
// init_reduce_max_min is only instantiated without the UInt16 mask (that format takes the manual column path).
static_assert(
    col_max_loadmacro_replay_len<false>() <= horizontal_stage_replay_start<HorizontalFold::MIN_MAX>(),
    "row MAX/MIN window overlaps the column LOADMACRO window");
static_assert(
    COL_MAX_MANUAL_REPLAY_LEN <= horizontal_stage_replay_start<HorizontalFold::MIN_MAX>(),
    "row MAX/MIN window overlaps the manual column MAX/MIN window");
static_assert(
    col_max_loadmacro_replay_len<true>() <= ckernel::math::replay_buf_offset &&
        COL_MAX_INT32_SIGNED_REPLAY_LEN <= ckernel::math::replay_buf_offset,
    "column windows must stay in the SFPU half of the replay buffer");

/**
 * @brief Records the butterfly-stage window. Called once from the init; never from a calculate.
 */
template <HorizontalFold fold>
inline void record_horizontal_reduce_stage() {
    lltt::record(horizontal_stage_replay_start<fold>(), horizontal_stage_replay_len<fold>());
    horizontal_reduce_rotate_first();
    horizontal_reduce_rotate_step();
    horizontal_reduce_fold<fold>();
}

/**
 * @brief Butterfly stages replayed from pieces of the [A, s, F] window: ror1 = A F (2 REPLAY), ror2 = A s F (1),
 *        ror4 = A s, s, s F (3). Each folds the accumulator with a copy of itself rotated by 1, 2 or 4 columns.
 */
template <HorizontalFold fold>
inline void horizontal_stage_ror1() {
    constexpr std::uint32_t start = horizontal_stage_replay_start<fold>();
    lltt::replay(start, HORIZONTAL_ROTATE_LEN);
    lltt::replay(start + 2 * HORIZONTAL_ROTATE_LEN, horizontal_fold_len<fold>());
}

template <HorizontalFold fold>
inline void horizontal_stage_ror2() {
    lltt::replay(horizontal_stage_replay_start<fold>(), horizontal_stage_replay_len<fold>());
}

template <HorizontalFold fold>
inline void horizontal_stage_ror4() {
    constexpr std::uint32_t start = horizontal_stage_replay_start<fold>();
    lltt::replay(start, 2 * HORIZONTAL_ROTATE_LEN);
    lltt::replay(start + HORIZONTAL_ROTATE_LEN, HORIZONTAL_ROTATE_LEN);
    lltt::replay(start + HORIZONTAL_ROTATE_LEN, HORIZONTAL_ROTATE_LEN + horizontal_fold_len<fold>());
}

/**
 * @brief The ror4, ror2, ror1 butterfly over the two accumulators, once per 8-row group: 6 REPLAY, 20 instructions
 *        (23 for the integer add). In: LREG0 / LREG4 = per-column partials of the two 4-row groups. Out: LREG0 / LREG4
 *        = the 8-column result in every column, so no trailing "move to column 0" rotate. Clobbers LREG1 / LREG5 only.
 */
template <HorizontalFold fold>
inline void horizontal_reduce_butterfly() {
    horizontal_stage_ror4<fold>();
    horizontal_stage_ror2<fold>();
    horizontal_stage_ror1<fold>();
}

/**
 * @brief Horizontal SUM of the two accumulators (see horizontal_reduce_butterfly).
 */
template <bool is_integer_mode>
inline void horizontal_reduce() {
    horizontal_reduce_butterfly<is_integer_mode ? HorizontalFold::ADD_INT : HorizontalFold::ADD_FLOAT>();
}

/**
 * @brief Horizontal MAX/MIN of the two accumulators (see horizontal_reduce_butterfly). Direction = SFPCONFIG bit 8 at
 *        replay time.
 */
inline void horizontal_reduce_max() { horizontal_reduce_butterfly<HorizontalFold::MIN_MAX>(); }

/**
 * @brief Sets the SFPSWAP(VEC_MIN_MAX) direction in one instruction. SFPCONFIG with instr_mod1 = 1 writes
 *        LaneConfig from the immediate (the form _init_sfpu_config_reg() uses), so it also resets the rest
 *        of the register and clobbers no LREG. Bit 8 clear = MAX on the row paths, bit 8 set = MIN.
 */
template <bool invert>
inline void set_sfpswap_direction() {
    TTI_SFPCONFIG((invert ? 0x0100 : 0x0000) /*imm16_math*/, 0xF /*config_dest*/, 1 /*instr_mod1*/);
}

/**
 * @brief SFPSWAP direction for the manual column MAX/MIN paths, which read the LOW (VD) operand of each
 *        compare-and-swap: MAX inverts (bit 8 set), MIN is the default, the opposite of the row paths.
 *
 * Called on entry by calculate_reduce_max_min_uint16 (UInt16 in a 32-bit Dest, and every non-Int32 format
 * under DISABLE_SFPLOADMACRO) and calculate_reduce_max_min_int32_col, instead of trusting the init: under one
 * shared init a preceding row MAX/MIN leaves the row convention in place (tt-metal#58551).
 */
template <PoolType pool_type>
inline void set_manual_col_swap_direction() {
    set_sfpswap_direction<pool_type == PoolType::MAX>();
}

/**
 * @brief Signed-Int32 horizontal MAX/MIN reduction (fully inline, no replay buffer).
 *
 * Mirrors horizontal_reduce_max()'s rotate-and-compare structure, but every compare-and-swap is the
 * two's-complement signed compare-and-swap (_emit_int32_signed_cswap_) so INT32_MIN is handled
 * correctly. The compare-and-swaps for the two parallel lanes (LREG0/1 and LREG4/5) are emitted
 * sequentially rather than interleaved because each signed compare-and-swap manages the condition-code
 * state (SFPSETCC/SFPENCC) and must not be interleaved with another. The SFPSHFT2 rotates keep the
 * original interleaved ordering (identical latency behavior up to each phase's compare-and-swap).
 */
inline void horizontal_reduce_max_int32() {
    // Phase 1: rotate by 4 and compare -> 8 values become 4 duplicated pair extrema.
    horizontal_reduce_rotate<4 /*shift*/>();
    _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _emit_int32_signed_cswap_<p_sfpu::LREG4, p_sfpu::LREG5>();

    // Phase 2: rotate by 2 and compare -> pair extrema become duplicated quad extrema.
    horizontal_reduce_rotate<2 /*shift*/>();
    _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _emit_int32_signed_cswap_<p_sfpu::LREG4, p_sfpu::LREG5>();

    // Phase 3: rotate by 1 and compare -> the full 8-column extreme in every column.
    horizontal_reduce_rotate<1 /*shift*/>();
    _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _emit_int32_signed_cswap_<p_sfpu::LREG4, p_sfpu::LREG5>();
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
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B or LO16).
 *         Int32 and UInt32 MAX/MIN do not use this helper; they take perform_reduce_row_max_tile_int32.
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in a 32-bit dest).
 * @param tile_row_offset Base row offset for this tile in the dest register
 * @param result_store_mode SFPSTORE mode for the per-row result (plain instruction mode, or mode 9 for UInt16 output)
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
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Vertical max: reduce left/right face pairs via compare-and-swap.
            // Two groups (LREG0-3 and LREG4-7) are interleaved so that each SFPSWAP
            // on independent registers hides the 2-cycle latency of the previous one.
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);  // hides LREG0/2 latency
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);  // hides LREG4/6 latency
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);  // hides LREG1/3 latency
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);  // hides LREG5/7 latency; LREG1 ready
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);  // hides LREG0/1 latency; LREG5 ready

            // Horizontal max: consolidate 8 SFPU columns into column 0
            horizontal_reduce_max();

            // result_store_mode is mode 9 only when this per-tile store is the final packer-visible
            // result (single column tile); otherwise it is intermediate and stays in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool is_avg = false>
inline void perform_reduce_row_sum_tile(
    std::uint32_t tile_row_offset,
    std::uint32_t result_store_mode,
    bool divide_now = false,
    RowAvgReciprocal recip = {}) {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Process tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31
    // Each face-pair iteration processes 8 rows (two groups of 4 rows each)
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        // Base offset for this face pair:
        // face_pair 0: faces 0+1 (dest rows 0-31)
        // face_pair 1: faces 2+3 (dest rows 32-63)
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            // Within each face, process rows in groups of 8 (two sub-groups of 4)
            std::uint32_t row_offset_first = row_group * 8;          // 0 or 8
            std::uint32_t row_offset_second = row_offset_first + 4;  // 4 or 12

            // Load 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            // Load next 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Replay the full tree-reduce buffer (both int and float use 6 instructions)
            lltt::replay(0, 6);

            // Horizontal reduction: consolidate all 8 SFPU columns into column 0 (interleaved for latency hiding)
            horizontal_reduce<is_integer_mode>();

            // For a single-column-tile AVG the per-tile sum is already the full row sum, so divide it
            // here (float-only path). When block_ct_dim > 1 the per-tile store is an intermediate
            // partial sum and the division is deferred to sum_first_columns_across_tiles instead.
            // The two results (first/second 4-row group) live in LREG0/LREG4; LREG2 is free after the
            // horizontal reduce and holds the reciprocal for the multiply.
            if constexpr (is_avg) {
                if (divide_now) {
                    load_row_avg_reciprocal_into(p_sfpu::LREG2, recip);
                    // The two SFPMULs cover each other's 2-cycle latency: the LREG4 multiply sits
                    // between the LREG0 multiply and the LREG0 store, and the first store sits
                    // between the LREG4 multiply and the LREG4 store, so no extra NOP is needed.
                    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
                }
            }

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is an intermediate store
            // re-loaded by the cross-tile accumulation and must stay in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
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
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16, bool is_avg = false>
inline void sum_first_columns_across_tiles(
    std::uint32_t tile_row_base, std::uint32_t block_ct_dim, RowAvgReciprocal recip = {}) {
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Row offset for each of the 8 partial row-sum results (face 0: 0,4,8,12; face 2: 32,36,40,44)
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        // Load tile 0's four LREGs at this batch's offsets (0,4,8,12 or 32,36,40,44) into LREG0-3
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);

        // Accumulate from remaining tiles
        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            // Load tile t's four LREGs at the same offsets into LREG4-7
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 3]);

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

        // LREG0-3 now hold the full per-row sum across all column tiles. For AVG (float-only path)
        // this is the point to divide by num_cols, before the final packer-visible store. LREG4 is
        // free after the accumulation loop and holds the reciprocal for the four multiplies.
        if constexpr (is_avg) {
            load_row_avg_reciprocal_into(p_sfpu::LREG4, recip);
            // The four SFPMULs cover each other's 2-cycle latency: at least two multiplies (and the
            // earlier stores) separate every multiply from the store that reads its result, so no
            // extra NOP is needed before the stores below.
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
            TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        }

        // Store LREG0-3 back to tile 0. This is the final, packer-visible result, so it uses mode 9
        // (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is UInt16 in a 32-bit dest (packer reads the
        // high 16 bits); a 32-bit output (e.g. UInt32) is stored with the plain INSTRUCTION_MODE.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0.
 *
 * Mirrors sum_first_columns_across_tiles but uses SFPSWAP instead of SFPADD
 * to keep maximum values rather than sums.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B or LO16).
 *         Int32 and UInt32 MAX/MIN do not use this helper; they take max_first_columns_across_tiles_int32.
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in a 32-bit dest).
 * @tparam pack_low16 Whether the final store uses mode 9 (low->high 16-bit) for a UInt16 output.
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void max_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 3]);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, 1);
        }

        // Final, packer-visible store: use mode 9 (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is
        // UInt16 in a 32-bit dest (pack_low16); a 32-bit output keeps the plain INSTRUCTION_MODE store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "perform_reduce_row_sum_avg only supports SUM and AVG pool types");

    // AVG divides each row sum by the number of reduced columns (32 per tile * block_ct_dim tiles).
    // The reciprocal is computed once and passed down to whichever step performs the final store.
    constexpr bool is_avg = (pool_type == PoolType::AVG);
    const RowAvgReciprocal recip = is_avg ? make_row_avg_reciprocal(32u * block_ct_dim) : RowAvgReciprocal{};

    // When there is a single column tile, the per-tile store is the final packer-visible result and must
    // use mode 9 only when the OUTPUT is UInt16 in a 32-bit dest (pack_low16). With multiple column tiles
    // the per-tile store is intermediate (re-loaded by sum_first_columns_across_tiles) and must stay in the
    // low 16 bits via INSTRUCTION_MODE; the final mode-9 (if any) is applied by the cross-tile store.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    // For AVG, the divide happens at the point the full row sum is known: in the per-tile reducer for a
    // single column tile, or in the cross-tile accumulation step for multiple column tiles.
    const bool divide_in_tile = is_avg && (block_ct_dim == 1);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        // Step 1: Reduce each tile individually (horizontal reduction within each tile)
        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_sum_tile<INSTRUCTION_MODE, clear_high_bits, is_avg>(
                tile_offset, tile_store_mode, divide_in_tile, recip);
        }

        // Step 2: Sum column 0 from all tiles in this row into tile 0's column 0 of this row
        if (block_ct_dim > 1) {
            sum_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16, is_avg>(
                tile_row_offset, block_ct_dim, recip);
        }
    }
}

/**
 * @brief Row-wise maximum/minimum reduction across a block of tiles.
 *
 * For each row of tiles, reduces every tile individually via perform_reduce_row_max_tile,
 * then (if block_ct_dim > 1) accumulates the per-tile column-0 extrema across tiles using
 * compare-and-swap into tile 0's column 0.
 *
 * MAX and MIN share the entire compare-and-swap machinery (perform_reduce_row_max_tile,
 * horizontal_reduce_max, max_first_columns_across_tiles); the only difference is the SFPSWAP
 * comparator direction, set once here via SFPCONFIG bit 8 (0 = MAX, 1 = MIN). The data
 * representation seen by SFPSWAP (float, or the non-negative sign-magnitude range for UInt16)
 * is identical for MAX and MIN, so flipping bit 8 cleanly inverts the result for every format that
 * reaches here. Int32 and UInt32 MAX/MIN do not use this function; they take the dedicated
 * two's-complement path (perform_reduce_row_max_min_int32 / calculate_reduce_max_min_int32_col).
 *
 * @tparam pool_type MAX or MIN.
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B or LO16)
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 * @param block_rt_dim Number of tiles along y axis of tensor (row tiles)
 */
template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_max_min(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "perform_reduce_row_max_min only supports MAX and MIN pool types");

    // Set the SFPSWAP direction on entry instead of trusting the init: in a multi-axis reduce (e.g. ttir.max
    // dim=[1,2]) one shared init serves both passes, and a manual column path run before this one leaves the
    // opposite direction in the config register. One SFPCONFIG resets the register and sets bit 8 for MIN.
    set_sfpswap_direction<pool_type == PoolType::MIN>();

    // Nothing is recorded here: the horizontal reduce replays the stage window init_reduce recorded above the
    // column window (see the slot map), so a column reduce after a row reduce under one shared init still works.

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
            perform_reduce_row_max_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
        }

        if (block_ct_dim > 1) {
            max_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Signed-Int32 per-tile row MAX/MIN reduction. Mirrors perform_reduce_row_max_tile but loads
 *        plain INT32 (bits preserved) and uses the signed compare-and-swap so INT32_MIN is handled.
 *
 * @param tile_row_offset Base row offset for this tile in the dest register
 * @param result_store_mode SFPSTORE mode for the per-row result (plain INT32 for the Int32 path)
 */
inline void perform_reduce_row_max_tile_int32(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
#pragma GCC unroll 2
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

#pragma GCC unroll 2
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            TT_SFPLOAD(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPLOAD(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first + 2);
            TT_SFPLOAD(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            TT_SFPLOAD(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            TT_SFPLOAD(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
            TT_SFPLOAD(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second + 2);
            TT_SFPLOAD(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            TT_SFPLOAD(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Vertical reduce via signed compare-and-swap (each is self-contained w.r.t. condition codes,
            // so unlike the float path they are emitted sequentially rather than interleaved).
            _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG2>();
            _emit_int32_signed_cswap_<p_sfpu::LREG4, p_sfpu::LREG6>();
            _emit_int32_signed_cswap_<p_sfpu::LREG1, p_sfpu::LREG3>();
            _emit_int32_signed_cswap_<p_sfpu::LREG5, p_sfpu::LREG7>();
            _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG1>();
            _emit_int32_signed_cswap_<p_sfpu::LREG4, p_sfpu::LREG5>();

            // Horizontal reduce: consolidate 8 SFPU columns into column 0.
            horizontal_reduce_max_int32();

            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Signed-Int32 cross-tile row MAX/MIN combine. Mirrors max_first_columns_across_tiles but loads
 *        plain INT32 and uses the signed compare-and-swap.
 *
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
inline void max_first_columns_across_tiles_int32(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 0]);
            TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 1]);
            TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 2]);
            TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 3]);

            _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG4>();
            _emit_int32_signed_cswap_<p_sfpu::LREG1, p_sfpu::LREG5>();
            _emit_int32_signed_cswap_<p_sfpu::LREG2, p_sfpu::LREG6>();
            _emit_int32_signed_cswap_<p_sfpu::LREG3, p_sfpu::LREG7>();
        }

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Signed-Int32 row MAX/MIN reduction across a block of tiles. Mirrors perform_reduce_row_max_min
 *        but routes through the signed-Int32 per-tile and cross-tile helpers (no recorded horizontal
 *        reduce buffer; horizontal_reduce_max_int32 is fully inline). Correct over the full Int32 range.
 *
 * @tparam pool_type MAX or MIN.
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 * @param block_rt_dim Number of tiles along y axis of tensor (row tiles)
 */
template <PoolType pool_type>
inline void perform_reduce_row_max_min_int32(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "perform_reduce_row_max_min_int32 only supports MAX and MIN pool types");

    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;

    // Re-establish the SFPSWAP direction (see perform_reduce_row_max_min): MAX is the default (bit 8 = 0),
    // MIN sets bit 8. The signed compare-and-swap's VEC_MIN_MAX obeys this bit; its both-negative fix is a
    // direction-independent unconditional exchange.
    set_sfpswap_direction<pool_type == PoolType::MIN>();

    // Int32 MAX/MIN never widens to a UInt16 output, so the per-tile store always uses the plain mode.
    const std::uint32_t tile_store_mode = static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_max_tile_int32(tile_offset, tile_store_mode);
        }

        if (block_ct_dim > 1) {
            max_first_columns_across_tiles_int32(tile_row_offset, block_ct_dim);
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
 * @brief Initialization for the manual (non-LOADMACRO) SFPU reduce MAX/MIN compare-and-swap path.
 *        Records a 3-instruction replay buffer that reduces LREG4-7 down to LREG4 via vertical
 *        compare-and-swaps. Used by the UInt16-in-32-bit-dest path (calculate_reduce_max_min_uint16) and by
 *        every non-Int32 format under DISABLE_SFPLOADMACRO, where the load and the swap are not fused into a
 *        LOADMACRO (for UInt16 the garbage high bits must be masked in between).
 *
 * @tparam INSTRUCTION_MODE The instruction mode (INT32 for UInt16 in 32-bit dest)
 * @tparam pool_type MAX or MIN; unused here. The direction is set on entry by the calculates
 *         (set_manual_col_swap_direction, set_sfpswap_direction) and read by the recorded swaps at replay time.
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min_int32() {
    _init_sfpu_config_reg();

    lltt::record(0, COL_MAX_MANUAL_REPLAY_LEN);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);
}

/**
 * @brief Init for the two's-complement signed Int32 column MAX/MIN reduce.
 *        Records a replay buffer that reduces LREG4-7 -> LREG4 via three signed compare-and-swaps (see
 *        _emit_int32_signed_cswap_). Used instead of the sign-magnitude (INT32_2S_COMP) path so INT32_MIN is
 *        handled correctly.
 *
 * @tparam pool_type MAX or MIN; unused here. The direction is set on entry by the calculates
 *         (calculate_reduce_max_min_int32_col, perform_reduce_row_max_min_int32).
 */
template <PoolType pool_type>
inline void init_reduce_max_min_int32_signed() {
    _init_sfpu_config_reg();

    lltt::record(0, COL_MAX_INT32_SIGNED_REPLAY_LEN);
    _emit_int32_signed_cswap_<p_sfpu::LREG7, p_sfpu::LREG6>();
    _emit_int32_signed_cswap_<p_sfpu::LREG6, p_sfpu::LREG5>();
    _emit_int32_signed_cswap_<p_sfpu::LREG5, p_sfpu::LREG4>();
}

/**
 * @brief Initialization for SFPU reduce MAX/MIN kernel.
 *        Sets up LOADMACRO sequences for compare-and-swap operations, configures address modifiers,
 *        and records replay buffers for efficient column-wise maximum/minimum reduction.
 *        For MIN operations, inverts the swap direction by configuring the SFPU control register.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for the float/unsigned formats that reach here: DEFAULT (FP32,
 * FP16B) or LO16. Int32 and UInt32 MAX/MIN use init_reduce_max_min_int32_signed, and UInt16 in a 32-bit Dest the manual
 * column path, instead, so it does not go through this init.
 * @tparam pool_type The PoolType enum value (MAX or MIN). MIN inverts the swap direction for minimum reduction.
 * @param num_cols The number of columns to process (typically 32 for a single tile, or multiple of 32 for block
 * operations)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type, bool clear_high_bits>
inline void init_reduce_max_min([[maybe_unused]] std::uint32_t num_cols) {
#ifdef DISABLE_SFPLOADMACRO
    init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
    return;
#endif

    // Reset the SFPU config and set the swap direction (MIN inverts) before defining the LOADMACRO sequences.
    // The LOADMACRO column path is the one path that trusts this init's direction.
    set_sfpswap_direction<pool_type == PoolType::MIN>();

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
    // MAX uses LOADMACRO mechanism.
    // When clearing high bits (UInt16 in 32-bit dest), two extra SFPAND ops mask the garbage high bits
    // of the manually loaded face-1 columns (LREG0/LREG1) before the compare-and-swap, so the buffer
    // grows from 9 to 11 instructions.
    lltt::record<lltt::NoExec>(0, col_max_loadmacro_replay_len<clear_high_bits>());
    TTI_INCRWC(0, 4, 0, 0);
    TTI_SFPLOADMACRO(5, INSTRUCTION_MODE, ADDR_MOD_3, 2);
    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 16);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 18);
    if constexpr (clear_high_bits) {
        TTI_SFPAND(0, CLEAR_REG, p_sfpu::LREG1, 0);
        TTI_SFPAND(0, CLEAR_REG, p_sfpu::LREG0, 0);
    }
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG0, 1);
    TTI_SFPLOADMACRO(0, INSTRUCTION_MODE, ADDR_MOD_3, 0);

    // Dummy loads to increment dest counters
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_2, 0);
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_1, 0);
}

/**
 * @brief Initialization for SFPU reduce SUM and AVG kernels, column and row.
 *        Records the column tree-add window and, above it, the row butterfly-stage window (see the slot map):
 *        Integer: 6 sequential instructions at [0, 6), first 3 reusable as half-reduce; row window [9, 16).
 *        Float: 6 interleaved instructions at [0, 6) + 4-slot half reduce at [6, 10); row window [10, 16).
 *        Interleaving eliminates NOPs in the full reduce by using B operations to hide A's 2-cycle latency.
 *
 * @note The row SUM/AVG calculate replays the row window instead of recording it, so call this init again after
 *       anything else records into the SFPU half of the replay buffer.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The PoolType enum value (SUM or AVG). AVG with a float format preloads the 1/32 constant.
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
    // the column AVG multiply collapses to a single SFPMUL instead of rebuilding the constant with two
    // SFPLOADI on every column group. Integer AVG uses perform_int_average (shift) and pays nothing; SUM
    // never averages, so it pays nothing either.
    if constexpr (pool_type == PoolType::AVG && !is_integer_mode) {
        sfpi::vConstFloatPrgm0 = 0.03125f;
    }

    if constexpr (is_integer_mode) {
        // Integer replay buffer: 6 sequential instructions (1-cycle SFPIADD latency, no NOPs needed)
        // Positions 0-2 form a natural half-reduce for LREG0-3 (reusable via replay(0, 3))
        lltt::record(0, COL_SUM_INT_REPLAY_LEN);

        // Upper face column summation (LREG0-3)
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 = LREG1 + LREG2
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = LREG0 + LREG1

        // Lower face column summation (LREG4-7)
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 = LREG6 + LREG7
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 = LREG5 + LREG6
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = LREG4 + LREG5
    } else {
        // Float replay buffer: 10 instructions total
        // Positions 0-5: Full reduce, interleaved to eliminate NOPs (B hides A's 2-cycle latency)
        // Positions 6-9: Half reduce for LREG0-3 only, with NOP for dependent SFPADD chain
        lltt::record(0, COL_SUM_FLOAT_REPLAY_LEN);

        // Full reduce (positions 0-5): interleaved upper/lower face summation
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);  // B1 (hides A1 latency)
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // B2 (hides A2 latency)
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // B3 (hides A3 latency)

        // Half reduce (positions 6-9): upper face only, NOP required for 2-cycle SFPADD latency
        // Reorders A1/A3/A2 so A1 and A3 (disjoint outputs) issue back-to-back; only A3's latency before A2 needs a
        // NOP.
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPNOP;  // Cover A3 latency (A1 is already available)
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);  // A2
    }

    // Row SUM/AVG butterfly-stage window, recorded once here (hoisted out of the per-tile row body).
    record_horizontal_reduce_stage<is_integer_mode ? HorizontalFold::ADD_INT : HorizontalFold::ADD_FLOAT>();
}

// ============================================================================
// Calculate Functions
// ============================================================================

/**
 * @brief Column-wise MAX/MIN reduction for UInt16 data in a 32-bit (fp32) dest, single 32x32 tile.
 *
 * The UInt16 datum lives in the low 16 bits of a 32-bit dest word with arbitrary garbage in the high
 * 16 bits, so every value must be masked (AND 0x0000FFFF) before it participates in a compare-and-swap.
 * The LOADMACRO pipeline used by calculate_reduce_max_min() fuses the load and the swap into a single
 * macro, leaving no place to clear the high bits in between; the garbage then dominates the integer
 * comparison and produces wrong minima/maxima for any non-constant input.
 *
 * Instead we use the same manual load/mask/swap structure as the Int32 path. Masked UInt16 values are
 * always non-negative, so a plain integer SFPSWAP orders them correctly (no sign-magnitude cast needed);
 * we move LREG0-3 -> LREG4-7 to feed the recorded swap buffer. Final, packer-visible results are written
 * with SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16) so the packer reads the low bits.
 *
 * Reuses init_reduce_max_min_int32()'s 3-swap replay buffer; the swap direction is set on entry. Only a single
 * 32x32 tile is processed per call, matching the column-reduce driver which invokes the kernel per tile.
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

    // This path reads the low operand of every compare-and-swap, so MAX needs the inverted direction. Set it
    // here rather than trusting the init: a row MAX/MIN under the same init leaves the opposite convention.
    set_manual_col_swap_direction<pool_type>();

    constexpr std::uint32_t ODD_COLUMNS = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2};  // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4] = {
        {0, 0, 32, 32},   // j=0: Face 0 and Face 2
        {16, 16, 48, 48}  // j=1: Face 1 and Face 3
    };

    // The intermediate stores below land in non-row-0 dest slots that we reload, so they use the
    // plain (full 32-bit) instruction mode. Only the final row-0 stores are packer-visible and use
    // mode 9 (SFPSTORE_MOD0_FMT_LO16) when the OUTPUT is UInt16 in a 32-bit dest (pack_low16), so the
    // packer reads the low 16 bits of the dest word.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    // Where each per-face partial (left in LREG4 by the reduce) is parked so it survives the remaining
    // face loads (which clobber LREG4-7). The order matches the LREG0-3 layout the transpose expects:
    // face i=0 (top even) -> LREG0, i=1 (top odd) -> LREG2, i=2 (bottom even) -> LREG1, i=3 (bottom odd) -> LREG3.
    constexpr std::uint32_t PARTIAL_LREG[4] = {p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3};

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = FACE_ADDRS[j][0];  // face 0 & 1 row-0 dst index

        // Reduce each of the four vertically adjacent faces (f0,f2 then f1,f3) within itself; the
        // max/min of its 16 rows is left in the top 4 rows. The masked face loads and the reduce replay
        // only touch LREG4-7, so each partial can be parked in LREG0-3 (via PARTIAL_LREG[i]) and survive
        // the remaining faces without a dest round-trip (a masked value stays masked through the swaps).
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            // Masked load straight into LREG4-7, where the recorded swap buffer reduces them.
            load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);

            lltt::replay(0, 3);  // compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPMOV(0, p_sfpu::LREG4, PARTIAL_LREG[i], 0);
        }

        // Move the four partials (now in LREG0-3) into LREG4-7 for the transpose + cross-row reduction.
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

        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, top_face_addr + ODD_COLUMNS);
    }
}

/**
 * @brief XOR bit 31 of every datum in the first @p num_tiles 32x32 tiles at the current DEST base. This maps
 *        UInt32 order onto two's-complement order and back, so UInt32 MAX/MIN can run on the signed Int32
 *        path. Uses LREG0 and LREG7.
 */
inline void flip_sign_bits(const std::uint32_t num_tiles) {
    constexpr std::uint32_t ODD_COLUMNS = 2;
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);  // 0x80000000
    for (std::uint32_t tile = 0; tile < num_tiles; tile++) {
        for (std::uint32_t row = 0; row < ROWS_PER_TILE; row += ROWS_PER_LOAD) {
            for (std::uint32_t column = 0; column <= ODD_COLUMNS; column += ODD_COLUMNS) {
                const std::uint32_t addr = tile * ROWS_PER_TILE + row + column;
                TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, addr);
                TTI_SFPXOR(0 /*imm12_math*/, p_sfpu::LREG7, p_sfpu::LREG0, 0 /*instr_mod1*/);
                TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, addr);
            }
        }
    }
}

/**
 * @brief Column-wise MAX/MIN reduction for signed Int32, single 32x32 tile, correct over the full
 *        Int32 range (including INT32_MIN).
 *
 * Same manual load/reduce/transpose structure as calculate_reduce_max_min_uint16, but the operands are
 * loaded with plain INT32 (two's-complement bits preserved) and every compare-and-swap is the signed
 * compare-and-swap (_emit_int32_signed_cswap_) instead of a bare sign-magnitude SFPSWAP. The vertical
 * LREG4-7 -> LREG4 reduction reuses the 3-swap replay buffer recorded by init_reduce_max_min_int32_signed
 * (3 * INT32_SIGNED_CSWAP_LEN instructions); the final face-pair combine emits the signed swap inline.
 *
 * Int32 MAX/MIN only supports a single tile (block_rt_dim == 1), which matches this single-tile kernel.
 *
 * @tparam pool_type MAX or MIN. MAX is the inverted swap direction here; it is set on entry, not taken from
 *         the init.
 * @tparam reduce_dim The reduction dimension; must be REDUCE_COL (column-wise) for this helper.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void calculate_reduce_max_min_int32_col() {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is supported here");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
    constexpr std::uint32_t REPLAY_LEN = 3 * INT32_SIGNED_CSWAP_LEN;

    // See calculate_reduce_max_min_uint16: the manual column path reads the low operand, so MAX inverts.
    set_manual_col_swap_direction<pool_type>();

    constexpr std::uint32_t ODD_COLUMNS = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2};  // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4] = {
        {0, 0, 32, 32},   // j=0: Face 0 and Face 2
        {16, 16, 48, 48}  // j=1: Face 1 and Face 3
    };

    // Where each per-face partial (left in LREG4 by the reduce) is parked so it survives the remaining
    // face loads (which clobber LREG4-7). The order matches the LREG0-3 layout the transpose expects:
    // face i=0 (top even) -> LREG0, i=1 (top odd) -> LREG2, i=2 (bottom even) -> LREG1, i=3 (bottom odd) -> LREG3.
    constexpr std::uint32_t PARTIAL_LREG[4] = {p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3};

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = FACE_ADDRS[j][0];  // face 0 & 1 row-0 dst index

        // Reduce each of the four vertically adjacent faces (f0,f2 then f1,f3) within itself; the
        // max/min of its 16 rows is left in the top 4 rows. The face loads and the reduce replay only
        // touch LREG4-7, so each partial can be parked in LREG0-3 (via PARTIAL_LREG[i]) and survive the
        // remaining faces without a dest round-trip.
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            load_face_data<INSTRUCTION_MODE, false, p_sfpu::LREG4>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);

            lltt::replay(0, REPLAY_LEN);  // signed compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPMOV(0, p_sfpu::LREG4, PARTIAL_LREG[i], 0);
        }

        // Move the four partials (now in LREG0-3) into LREG4-7 for the transpose + cross-row reduction.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

        // Transpose so the 4 partial results of each column sit in one register, reduce, transpose back.
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(0, REPLAY_LEN);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Signed compare-and-swap to combine the two vertically adjacent faces (even and odd columns).
        _emit_int32_signed_cswap_<p_sfpu::LREG7, p_sfpu::LREG6>();  // odd columns of face pair
        _emit_int32_signed_cswap_<p_sfpu::LREG5, p_sfpu::LREG4>();  // even columns of face pair

        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, top_face_addr + ODD_COLUMNS);
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
 *        - For MIN operations, swap direction is inverted (configured during init via SFPU control register)
 *        - Transposes and sorts results to align maxima/minima correctly across LREG4-7
 *        - Stores final maximum/minimum values to row 0 (32 datums across faces 0 and 1)
 *
 * @tparam pool_type The PoolType enum value (MAX or MIN). MIN uses inverted swap direction for minimum reduction.
 * @tparam reduce_dim The reduction dimension: REDUCE_COL for column-wise, REDUCE_ROW for row-wise (MAX and MIN).
 * @tparam INSTRUCTION_MODE The instruction mode for the float/unsigned formats that reach here: DEFAULT (FP32,
 * FP16B) or LO16. Int32 and UInt32 MAX/MIN use the dedicated two's-complement path
 * (calculate_reduce_max_min_int32_col / perform_reduce_row_max_min_int32), and UInt16 in a 32-bit Dest the manual
 * column path (calculate_reduce_max_min_uint16), instead.
 * @param block_ct_dim Number of tiles along x axis (column tiles, default 1).
 * @param block_rt_dim Number of tiles along y axis (row tiles, default 1).
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min(
    [[maybe_unused]] const std::uint32_t block_ct_dim = 1, [[maybe_unused]] const std::uint32_t block_rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            ((pool_type == PoolType::MAX || pool_type == PoolType::MIN) && reduce_dim == ReduceDim::REDUCE_ROW),
        "Only column reduction (REDUCE_COL) and row MAX/MIN reduction (REDUCE_ROW with MAX/MIN) are supported");

#ifdef DISABLE_SFPLOADMACRO
    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        // LOADMACRO-disabled builds use the manual load/swap column reducer instead of recording
        // SFPLOADMACRO instructions in the generic column MAX/MIN path.
        calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
        return;
    }
#endif

    if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        static_assert(
            pool_type == PoolType::MAX || pool_type == PoolType::MIN,
            "Row MAX/MIN reduction (REDUCE_ROW) only supports MAX and MIN pool types here");
        perform_reduce_row_max_min<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
            block_ct_dim, block_rt_dim);
    } else {
        // The recorded replay buffer in init_reduce_max_min has two extra SFPAND ops when clearing
        // high bits, so the per-face-pair replay window grows accordingly.
        constexpr std::uint32_t replay_buffer_offset = clear_high_bits ? 9 : 7;
        constexpr std::uint32_t replay_buffer_next_face = clear_high_bits ? 10 : 8;

        // Initial loads: LREG4-7 will hold maximum values across F0 and F1
        TTI_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        TTI_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, 2);
        TTI_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, 16);
        TTI_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, 18);

        // First tile processing (F0, F1, F2, F3)
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face);

        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face + 1);

        // Remaining tiles
        for (std::uint32_t i = 0; i < block_rt_dim - 1; i++) {
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
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG6, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Store results to first row.
        // For UInt16 OUTPUT in a 32-bit dest the reduced value lives in the low 16 bits of the LREG, but
        // the packer reads the high 16 bits of the dest word. SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16)
        // writes the low 16 bits into the half the packer consumes; a 32-bit output keeps the plain store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TTI_SFPSTORE(p_sfpu::LREG4, STORE_MODE, ADDR_MOD_3, 0);
        TTI_SFPSTORE(p_sfpu::LREG5, STORE_MODE, ADDR_MOD_3, 2);
        TTI_SFPSTORE(p_sfpu::LREG6, STORE_MODE, ADDR_MOD_3, 16);
        TTI_SFPSTORE(p_sfpu::LREG7, STORE_MODE, ADDR_MOD_3, 18);
    }
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
 *        For AVG mode: signed Int32 divides by 32 with a magnitude, logical shift and sign restore; the
 *        unsigned formats (UInt32, UInt16) use a plain logical shift; float formats multiply by 1/32.
 *
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG)
 * @tparam reduce_dim REDUCE_COL (all formats) or REDUCE_ROW (SUM all formats; AVG float formats only)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam is_signed_int True when the reduce format is signed (Int32). Passed separately from INSTRUCTION_MODE
 * because UInt32 shares the INT32 mode; see perform_int_average().
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void calculate_reduce_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "Only SUM and AVG pool types are currently supported on SFPU");

    // Integer vs float is determined by the load/store mode. Row AVG divides the row sum by the
    // (runtime) column count, which is only exact via a reciprocal multiply for float formats; an
    // integer row AVG by an arbitrary column count would need a general integer divide and is not
    // supported (integer AVG stays column-only, where the divisor is the fixed 32 rows of a tile).
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Compile-time assertions to restrict to currently supported operations
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || (pool_type == PoolType::AVG && !is_integer_mode))),
        "Row reduction (REDUCE_ROW) supports SUM (all formats) and AVG (float formats only)");

    // Supported instruction modes for SFPU reduce sum/avg (integer and float)
    constexpr bool is_supported_reduce_instr_mode =
        (is_integer_mode || INSTRUCTION_MODE == InstrModLoadStore::DEFAULT ||
         INSTRUCTION_MODE == InstrModLoadStore::FP32 || INSTRUCTION_MODE == InstrModLoadStore::FP16B);
    static_assert(
        is_supported_reduce_instr_mode,
        "INSTRUCTION_MODE must be one of: INT32, INT32_2S_COMP, LO16, DEFAULT, FP32, FP16B");

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        perform_reduce_col_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int>();
    } else {
        perform_reduce_row_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
            block_ct_dim, block_rt_dim);
    }
    // For column reductions: sums are stored horizontally in the first row of tensor in dest reg
    // For row reductions: sums are stored vertically in the first column of tensor in dest reg
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Unified reduction init kernel wrapper for SFPU reduce kernel.
 *        Determines the instruction mode from format, then dispatches to the appropriate init kernel. One init serves
 *        REDUCE_COL and REDUCE_ROW: it records the column window and, for the row paths that replay one, the row
 *        butterfly-stage window above it, all in the SFPU half [0, 16) of the replay buffer (see the slot map).
 * @note The row SUM/AVG/MAX/MIN calculates replay what this init recorded, so call it again after anything else
 *       records into the SFPU half of the replay buffer.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX)
 * @tparam format The data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b)
 * @param block_ct_dim Block dimension (used for MAX reduction to specify number of columns, default is 1 for single
 * tile)
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_accum_en>
inline void init_reduce(std::uint32_t block_ct_dim = 1) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Int32 DEST holds two's-complement for every reduction.
    // SUM and AVG load with plain INT32 so the word reaches SFPIADD (a two's-complement adder) unchanged;
    // INT32_2S_COMP applies SignMagToTwosComp on load and would corrupt two's-complement negatives.
    // Int32 MAX/MIN dispatch to init_reduce_max_min_int32_signed below (plain INT32 load + a software
    // signed compare-and-swap, correct over the full range including INT32_MIN), so they do not use
    // INSTRUCTION_MODE here.
    constexpr InstrModLoadStore INSTRUCTION_MODE = (format == DataFormat::Float16_b)
                                                       ? InstrModLoadStore::DEFAULT
                                                       : GetSfpLoadStoreInstrMod<format, is_fp32_dest_accum_en>();

    // Garbage high bits need to be cleared when loading UInt16 data from a 32-bit (fp32) dest word.
    constexpr bool clear_high_bits = (is_fp32_dest_accum_en && format == DataFormat::UInt16);

    // Signed Int32 MAX/MIN uses a dedicated two's-complement compare-and-swap path (correct over the
    // full Int32 range, including INT32_MIN) instead of the sign-magnitude INT32_2S_COMP path. The
    // column reduce consumes the LREG4-7 -> LREG4 replay buffer recorded by init_reduce_max_min_int32_signed
    // below; the row reduce takes a separate helper whose horizontal reduce is fully inline (no re-record).
    // UInt32 MAX/MIN takes the same path: calculate_reduce flips bit 31 around it.
    constexpr bool int32_max_min =
        ((format == DataFormat::Int32 || format == DataFormat::UInt32) &&
         (pool_type == PoolType::MAX || pool_type == PoolType::MIN));

    if constexpr (clear_high_bits) {
        // CLEAR_REG (sfpi::vConstIntPrgm0 / LREG12) holds the mask used by SFPLOAD_EXT to clear high bits.
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }

    // Dispatch to appropriate PoolType init
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (int32_max_min) {
            init_reduce_max_min_int32_signed<pool_type>();
        } else if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest uses the manual (non-LOADMACRO) compare-and-swap path so the
            // garbage high bits can be masked before each swap. It records the manual 3-swap replay
            // buffer; the calculate sets the swap direction on entry. That format is column-only
            // (calculate_reduce_max_min_uint16), so no row window is recorded.
            init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
        } else {
            init_reduce_max_min<INSTRUCTION_MODE, pool_type, false>(block_ct_dim);
            // Hoisted out of perform_reduce_row_max_min: the row butterfly-stage window is recorded once here.
            // (Int32/UInt32 row MAX/MIN keeps its inline signed compare-and-swap reduce and records nothing.)
            record_horizontal_reduce_stage<HorizontalFold::MIN_MAX>();
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        init_reduce_sum_avg<INSTRUCTION_MODE, pool_type>();
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX");
    }
}

/**
 * @brief Unified reduction kernel wrapper for a 32x32 tile.
 *        Determines the instruction mode from format, then dispatches to the appropriate reduction kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam reduce_dim REDUCE_COL (all pools/formats) or REDUCE_ROW (SUM/MAX/MIN all formats; AVG float formats only)
 * @tparam format The INPUT data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b). Drives the
 *         instruction mode and load-time high-bit masking.
 * @tparam output_format The packer-visible OUTPUT data format (defaults to @p format). Drives the final store mode:
 *         UInt16 output in a 32-bit dest is stored via mode 9 (low->high 16-bit swap), while a 32-bit output (e.g.
 *         UInt32) is stored with the plain instruction mode. This lets UInt16 input be summed into a UInt32 output
 *         without overflow.
 * @param block_ct_dim Block dimension (used for SUM/AVG column reduction to specify number of columns, default is 1 for
 * single tile)
 * @param block_rt_dim Block dimension (used for MAX/MIN reduction to specify block height, or SUM row reduction;
 * default is 1 for single tile)
 *
 * @note Constraints (unable to static assert for block_rt_dim runtime parameter)
 *       - Column MAX/MIN with Int32 or UInt32 format only supports block_ct_dim == block_rt_dim == 1 (single tile)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    bool is_fp32_dest_accum_en,
    DataFormat output_format = format>
inline void calculate_reduce(std::uint32_t block_ct_dim = 1, std::uint32_t block_rt_dim = 1) {
    // Row reduction supports SUM/MAX/MIN for every supported format; AVG is row-supported only for
    // float formats, because the row divisor is the runtime column count and only the float
    // reciprocal-multiply divides exactly (integer AVG stays column-only with its fixed /32 divisor).
    constexpr bool is_float_format = (format == DataFormat::Float32 || format == DataFormat::Float16_b);
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || pool_type == PoolType::MAX || pool_type == PoolType::MIN ||
              (pool_type == PoolType::AVG && is_float_format))),
        "Row reduction (REDUCE_ROW) supports SUM/MAX/MIN (all formats) and AVG (float formats only)");
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Int32 load/store mode per reduction. DEST holds two's-complement Int32 for every reduction.
    //   SUM (two's-complement): plain INT32 so SFPIADD (a two's-complement adder) gets the word unchanged;
    //        INT32_2S_COMP applies SignMagToTwosComp on load and would corrupt negatives.
    //   MAX/MIN (two's-complement): plain INT32 (bits preserved), feeding the software signed
    //        compare-and-swap path (see int32_max_min below), which is correct over the full Int32 range
    //        including INT32_MIN. This applies to BOTH the column reduce and the row reduce, so a
    //        multi-axis reduce (e.g. ttir.max dim=[1,2]) that chains column-then-row over the same DEST
    //        stays consistent (both axes leave/consume plain two's-complement).
    //   AVG (two's-complement): plain INT32, like SUM, so the column sum and perform_int_average's
    //        divide-by-32 see the word unchanged.
    //   UInt32 MAX/MIN: the same signed path, with bit 31 of every datum flipped before and after, which maps
    //        unsigned order onto two's-complement order (SFPSWAP's sign-magnitude order would rank words
    //        with bit 31 set below the rest).
    constexpr bool uint32_max_min =
        (format == DataFormat::UInt32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN));
    constexpr bool int32_max_min =
        uint32_max_min || (format == DataFormat::Int32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN));
    // Int32 and UInt32 MAX/MIN (both column and row) use a dedicated two's-complement compare-and-swap path
    // that loads with plain INT32 (see the per-reduction note above).
    constexpr InstrModLoadStore INSTRUCTION_MODE = int32_max_min ? InstrModLoadStore::INT32
                                                   : (format == DataFormat::Float16_b)
                                                       ? InstrModLoadStore::DEFAULT
                                                       : GetSfpLoadStoreInstrMod<format, is_fp32_dest_accum_en>();

    // Signedness of the data, which INSTRUCTION_MODE does not carry: Int32, UInt32 and UInt16 in a 32-bit
    // dest all load with INT32. Only the integer column AVG divide consumes it.
    constexpr bool is_signed_int = (format == DataFormat::Int32);

    // Garbage high bits need to be cleared when loading UInt16 data from a 32-bit (fp32) dest word
    // (driven by INPUT format).
    constexpr bool clear_high_bits = (is_fp32_dest_accum_en && format == DataFormat::UInt16);

    // The packer-visible result must go through mode-9 (low->high 16-bit) store only when the OUTPUT is UInt16
    // in a 32-bit dest. A 32-bit output (e.g. UInt32) keeps the full word, so it uses the plain store. This is
    // driven by the OUTPUT format and is independent of the load-time masking above.
    constexpr bool pack_low16 = (is_fp32_dest_accum_en && output_format == DataFormat::UInt16);
    static_assert(
        !(int32_max_min && pack_low16),
        "Int32/UInt32 MAX/MIN stores the full 32-bit word; a UInt16 output (low-16 store) is not supported");

    // Dispatch to appropriate reduction kernel based on PoolType
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (int32_max_min) {
            // Int32 (and bit-31-flipped UInt32) MAX/MIN: two's-complement compare-and-swap path (handles INT32_MIN).
            [[maybe_unused]] const std::uint32_t num_tiles = block_ct_dim * block_rt_dim;
            if constexpr (uint32_max_min) {
                flip_sign_bits(num_tiles);
            }
            if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
                // The column kernel is single-tile (32x32) and ignores block_ct_dim/block_rt_dim, so guard the
                // single-tile contract loudly rather than silently dropping tiles if a caller passes >1.
                LLK_ASSERT(
                    block_ct_dim == 1 && block_rt_dim == 1,
                    "Int32/UInt32 column MAX/MIN reduce is single-tile (block_ct_dim == block_rt_dim == 1)");
                calculate_reduce_max_min_int32_col<pool_type, reduce_dim>();
            } else {
                static_assert(
                    reduce_dim == ReduceDim::REDUCE_ROW,
                    "Int32/UInt32 MAX/MIN reduce supports REDUCE_COL and REDUCE_ROW");
                perform_reduce_row_max_min_int32<pool_type>(block_ct_dim, block_rt_dim);
            }
            if constexpr (uint32_max_min) {
                flip_sign_bits(num_tiles);
            }
        } else if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest: manual load/mask/swap path (LOADMACRO cannot mask between load and swap).
            calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
        } else {
            calculate_reduce_max_min<pool_type, reduce_dim, INSTRUCTION_MODE, false, pack_low16>(
                block_ct_dim, block_rt_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        calculate_reduce_sum_avg<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int>(
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
