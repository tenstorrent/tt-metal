// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// Dest geometry
// ============================================================================
// A 32x32 Dest tile is 64 addr units of 16 datums, one per face row: face f holds units
// 16f to 16f+15, so the tile's four faces sit at 0, 16, 32 and 48. An SFPLOAD covers the
// four units of [addr & ~3, +3] and, by address bit 1, either the even or the odd 8 datums
// of each - 4 face rows x 8 datums = the SFPU's 4 rows x 8 column instances.
//
// That gives the two units this file addresses Dest in:
//   a "column group" - one (face, column parity) pair, 16 face rows deep, and
//   a "row quad"     - four whole tile rows, i.e. both parities of two side-by-side faces.

constexpr std::uint32_t REDUCE_FACE_STRIDE = FACE_R_DIM;                   // 16 addr units per face
constexpr std::uint32_t REDUCE_FACE_PAIR_STRIDE = 2 * REDUCE_FACE_STRIDE;  // faces 0/1 then faces 2/3
constexpr std::uint32_t REDUCE_TILE_STRIDE = 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
constexpr std::uint32_t REDUCE_ROWS_PER_LOAD = 4;  // face rows one SFPLOAD covers
constexpr std::uint32_t REDUCE_QUADS_PER_FACE_PAIR = FACE_R_DIM / REDUCE_ROWS_PER_LOAD;
constexpr std::uint32_t REDUCE_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t REDUCE_SFPU_COLUMNS = 8;         // SFPU column instances a row spans
constexpr std::uint32_t REDUCE_COL_EXTENT = TILE_R_DIM;  // rows a column reduce collapses
constexpr std::uint32_t REDUCE_COLS_PER_TILE = TILE_C_DIM;

// Working registers. Only LREG0-7 may be written: on Quasar an LREG destination index of
// 12-15 captures the instruction into a Load Macro register instead of executing it, and
// 9/10/11 are the read-only constants 0.0, 1.0 and -1.0.
constexpr std::uint32_t REDUCE_ACC_REG = p_sfpu::LREG4;    // row reduce: running per-row total
constexpr std::uint32_t REDUCE_ROT_REG = p_sfpu::LREG5;    // row reduce: rotated copy of the accumulator
constexpr std::uint32_t REDUCE_RECIP_REG = p_sfpu::LREG6;  // row AVG: 1/num_cols, live for the whole block

// bfloat16 encoding of 1/REDUCE_COL_EXTENT, the column-AVG divisor. A power of two, so the
// SFPMULI that applies it is exact.
constexpr std::uint32_t REDUCE_RECIP_COL_EXTENT_FP16B = 0x3D00;
static_assert(REDUCE_COL_EXTENT == 32, "REDUCE_RECIP_COL_EXTENT_FP16B is the bfloat16 encoding of 1/32");

// SFPSETCC imm12 bit 11: interpret src_c as two's-complement INT32 rather than FP32/SMAG32,
// which turns "less than zero" into a plain sign-bit test. Same encoding the Quasar
// element-wise max/min kernel uses (@ref calculate_binary_max_min).
constexpr std::uint32_t REDUCE_SFPSETCC_INT32_SIGNBIT = 0x800;

// SFPSHFT2 instr_mod1 3: rotate one LREG per row across the SFPU column instances, column X
// feeding column X+1 and wrapping. The only instruction that moves data between columns, and
// so the only way a row reduce can fold its 8 column instances together.
constexpr std::uint32_t REDUCE_SFPSHFT2_ROTATE = 3;

/**
 * @brief SFPLOAD/SFPSTORE format-select code for a reduce operand format.
 *
 * Int32 names its mode explicitly because implied formats are unreliable under unpack-to-dest
 * (Quasar TEN-4674). Every float format takes DEFAULT, which resolves the Dest word format at
 * runtime from ALU_ACC_CTRL_SFPU_Fp32 and the SrcB format register - already programmed from
 * formats.math - so one kernel body covers Float32, Float16_b and Float16 and both Dest widths.
 *
 * @tparam FORMAT: Math-side data format of the reduce operand
 */
template <DataFormat FORMAT>
inline constexpr std::uint32_t reduce_sfpmem_mode() {
    return (FORMAT == DataFormat::Int32) ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;
}

/** @brief Whether @p FORMAT reduces on the integer instruction path (SFPIADD, no float scaling). */
template <DataFormat FORMAT>
inline constexpr bool reduce_is_int_format() {
    return FORMAT == DataFormat::Int32;
}

/** @brief Formats this kernel implements. */
constexpr bool is_supported_reduce_format(DataFormat format) {
    return format == DataFormat::Float32 || format == DataFormat::Float16_b || format == DataFormat::Float16 ||
           format == DataFormat::Int32;
}

// ============================================================================
// Pairwise combine
// ============================================================================

/**
 * @brief Fold @p SRC into @p DST with the reduction's operator, leaving the result in @p DST.
 *
 * SUM and AVG accumulate; AVG only differs from SUM in the scaling its caller applies once the
 * whole extent is folded. MAX compares with SFPSWAP, which places the larger operand in lreg_c
 * and the smaller in lreg_dest - so passing @p DST as lreg_c is what makes @p DST the maximum.
 *
 * SFPSWAP compares its operands as two's-complement int32. That is exactly right for Int32,
 * whose Dest words are already two's-complement. IEEE float bits are sign-magnitude, and a
 * two's-complement comparator orders them correctly unless *both* operands are negative, where
 * the magnitude ordering inverts; the CC-guarded re-swap corrects precisely those lanes. The
 * float path therefore costs a comparison and a correction, the Int32 path only a comparison.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam IS_INT: Whether the operands are integers (skips the float sign-magnitude correction)
 * @tparam DST: LREG holding the running result; overwritten with the folded value
 * @tparam SRC: LREG holding the incoming value; clobbered
 * @note SFPSWAP is a 2-cycle op whose result must not be consumed by the very next instruction
 *       (the SFPSWAP -> SFPSTORE auto-stall bug), so every MAX fold ends in an SFPNOP. That
 *       makes this callable back-to-back and directly ahead of a store.
 */
template <PoolType POOL_TYPE, bool IS_INT, std::uint32_t DST, std::uint32_t SRC>
inline void reduce_combine() {
    static_assert(DST <= p_sfpu::LREG7 && SRC <= p_sfpu::LREG7, "reduce may only write LREG0-7");
    static_assert(DST != SRC, "reduce_combine needs two distinct registers");

    if constexpr (POOL_TYPE == PoolType::MAX) {
        // Two's-complement compare: DST <- larger, SRC <- smaller.
        TTI_SFPSWAP(0 /* imm12 */, DST, SRC, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

        if constexpr (!IS_INT) {
            // Successive SFPSETCCs AND into CC, so the swap below fires only where both
            // operands are negative - the lanes the two's-complement compare ordered backwards.
            TTI_SFPSETCC(REDUCE_SFPSETCC_INT32_SIGNBIT, DST, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(REDUCE_SFPSETCC_INT32_SIGNBIT, SRC, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSWAP(0 /* imm12 */, DST, SRC, p_sfpswap::UNCONDITIONALLY);
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        }
    } else if constexpr (IS_INT) {
        // SFPIADD raises CC off the result sign by default; the reduce never predicates on it.
        TTI_SFPIADD(0 /* imm12 */, SRC, DST, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        // SFPADD is dest = a * b + c, so a = DST and b = 1.0 makes it dest = DST + SRC.
        TTI_SFPADD(DST, p_sfpu::LCONST_1, SRC, DST, 0 /* instr_mod1: no negation */);
    }
}

// ============================================================================
// Column reduce (REDUCE_COL): collapse a tile's 32 rows onto row 0
// ============================================================================

/**
 * @brief Reduce one column group of a tile - 8 of its 32 columns, over all 32 rows.
 *
 * A column group is one column parity of one top face and of the bottom face directly below it,
 * which between them cover those 8 columns across every row of the tile. Eight SFPLOADs bring in
 * all 32 rows; the two banks tree-reduce to LREG0 and LREG4, which fold together into a lane
 * (r, c) holding rows {r, r+4, r+8, r+12} of both faces at column c. SFPTRANSP then lifts LREG0's
 * four rows into row 0 of LREG0-3, where a second fold finishes the column.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam MODE: SFPLOAD/SFPSTORE format-select code, @ref reduce_sfpmem_mode
 * @tparam IS_INT: Whether operands are integers
 * @tparam TOP_FACE_ADDR: Dest address of the top face of the pair, values = <0/REDUCE_FACE_STRIDE>
 * @tparam COLUMN_OFFSET: Column parity selector, values = <EVEN_COL/ODD_COL>
 * @note Writes the group's totals to row 0 of the top face and leaves rows 1-3 holding the
 *       transpose's leftovers, which is all a column reduce's consumers read.
 */
template <PoolType POOL_TYPE, std::uint32_t MODE, bool IS_INT, std::uint32_t TOP_FACE_ADDR, std::uint32_t COLUMN_OFFSET>
inline void reduce_col_group() {
    constexpr std::uint32_t TOP = TOP_FACE_ADDR + COLUMN_OFFSET;
    constexpr std::uint32_t BOT = TOP + REDUCE_FACE_PAIR_STRIDE;

    // Top face rows 0-15 into LREG0-3, the face below it into LREG4-7.
    TTI_SFPLOAD(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, TOP + 0 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG1, MODE, ADDR_MOD_7, 0 /* done */, TOP + 1 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG2, MODE, ADDR_MOD_7, 0 /* done */, TOP + 2 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG3, MODE, ADDR_MOD_7, 0 /* done */, TOP + 3 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG4, MODE, ADDR_MOD_7, 0 /* done */, BOT + 0 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG5, MODE, ADDR_MOD_7, 0 /* done */, BOT + 1 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG6, MODE, ADDR_MOD_7, 0 /* done */, BOT + 2 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG7, MODE, ADDR_MOD_7, 0 /* done */, BOT + 3 * REDUCE_ROWS_PER_LOAD);

    // Fold each bank down to its first register. The two chains are independent, so interleaving
    // them keeps a dependent pair from ever landing back-to-back.
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG6, p_sfpu::LREG7>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG1, p_sfpu::LREG2>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG5, p_sfpu::LREG6>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG4, p_sfpu::LREG5>();

    // Both faces together: every lane of LREG0 now carries a quarter of its column.
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG4>();

    // The four quarters sit in LREG0's four rows, which no instruction can fold in place.
    // SFPTRANSP redistributes them to row 0 of LREG0-3, where the ordinary combine reaches them.
    TTI_SFPTRANSP;
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG1, p_sfpu::LREG2>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();

    if constexpr (POOL_TYPE == PoolType::AVG) {
        // A column reduce always spans the tile's 32 rows, so the divisor is a compile-time
        // power of two and rides along as an SFPMULI immediate - no constant register needed.
        TTI_SFPMULI(REDUCE_RECIP_COL_EXTENT_FP16B, p_sfpu::LREG0, 0 /* instr_mod1 */);
    }

    TTI_SFPSTORE(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, TOP);
}

/**
 * @brief Column-reduce one whole 32x32 Dest tile in place.
 *
 * Walks the tile's four column groups - both parities of the two top faces - which between them
 * cover all 32 columns of row 0.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 */
template <PoolType POOL_TYPE, DataFormat FORMAT>
inline void reduce_col_tile() {
    constexpr std::uint32_t MODE = reduce_sfpmem_mode<FORMAT>();
    constexpr bool IS_INT = reduce_is_int_format<FORMAT>();
    constexpr std::uint32_t FACE_1 = REDUCE_FACE_STRIDE;

    reduce_col_group<POOL_TYPE, MODE, IS_INT, 0, p_sfpu::col_offset::EVEN_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, 0, p_sfpu::col_offset::ODD_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, FACE_1, p_sfpu::col_offset::EVEN_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, FACE_1, p_sfpu::col_offset::ODD_COL>();
}

// ============================================================================
// Row reduce (REDUCE_ROW): collapse a tile row's columns onto column 0
// ============================================================================

/**
 * @brief Fold the accumulator's 8 SFPU column instances together, in place.
 *
 * SFPU columns are independent lanes; SFPSHFT2's rotate is the only path between them. Three
 * rotate-and-fold stages - by 4, then 2, then 1 - leave the total of all eight in every column,
 * so the store that follows writes it to tile column 0 whichever lane the packer reads.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam IS_INT: Whether operands are integers
 * @note Each rotate is separated from its consumer by an SFPNOP; SFPSHFT2 is a 2-cycle op and
 *       this kernel does not rely on the pipeline interlocking a dependent reader.
 */
template <PoolType POOL_TYPE, bool IS_INT>
inline void reduce_row_fold_columns() {
#pragma GCC unroll 1
    for (std::uint32_t distance = REDUCE_SFPU_COLUMNS / 2; distance > 0; distance >>= 1) {
        TTI_SFPMOV(REDUCE_ACC_REG, REDUCE_ROT_REG, 0 /* instr_mod1: plain copy */);
        for (std::uint32_t step = 0; step < distance; step++) {
            TTI_SFPSHFT2(0 /* imm12 */, REDUCE_ROT_REG, REDUCE_ROT_REG, REDUCE_SFPSHFT2_ROTATE);
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        }
        reduce_combine<POOL_TYPE, IS_INT, REDUCE_ACC_REG, REDUCE_ROT_REG>();
    }
}

/**
 * @brief Row-reduce a block of tiles held in Dest, writing each row's total to its column 0.
 *
 * Walks Dest one row quad at a time - four whole tile rows, addressed as both column parities of
 * two side-by-side faces. For each quad the accumulator sweeps the tile row's @p block_ct_dim
 * tiles, after which the quad's 8 SFPU columns fold together and the total lands in column 0 of
 * the tile row's first tile.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 * @param block_ct_dim: Tiles per tile row, i.e. the width being reduced
 * @param block_rt_dim: Tile rows in the block
 * @note Every tile of the block must already be in Dest; unlike the column reduce this cannot be
 *       driven a block at a time, because a row's total spans the whole tile row.
 * @note AVG expects @ref reduce_row_load_reciprocal to have put 1/num_cols in REDUCE_RECIP_REG,
 *       which this leaves untouched.
 */
template <PoolType POOL_TYPE, DataFormat FORMAT>
inline void reduce_row_block(const std::uint32_t block_ct_dim, const std::uint32_t block_rt_dim) {
    constexpr std::uint32_t MODE = reduce_sfpmem_mode<FORMAT>();
    constexpr bool IS_INT = reduce_is_int_format<FORMAT>();

    for (std::uint32_t rt = 0; rt < block_rt_dim; rt++) {
        const std::uint32_t row_base = rt * block_ct_dim * REDUCE_TILE_STRIDE;

        for (std::uint32_t face_pair = 0; face_pair < REDUCE_FACE_PAIRS; face_pair++) {
            for (std::uint32_t quad = 0; quad < REDUCE_QUADS_PER_FACE_PAIR; quad++) {
                // Four tile rows: both parities of the left face and of the face beside it.
                const std::uint32_t quad_offset = (face_pair * REDUCE_FACE_PAIR_STRIDE) + (quad * REDUCE_ROWS_PER_LOAD);

                for (std::uint32_t ct = 0; ct < block_ct_dim; ct++) {
                    const std::uint32_t tile_base = row_base + (ct * REDUCE_TILE_STRIDE) + quad_offset;

                    TT_SFPLOAD(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, tile_base + p_sfpu::col_offset::EVEN_COL);
                    TT_SFPLOAD(p_sfpu::LREG1, MODE, ADDR_MOD_7, 0 /* done */, tile_base + p_sfpu::col_offset::ODD_COL);
                    TT_SFPLOAD(
                        p_sfpu::LREG2,
                        MODE,
                        ADDR_MOD_7,
                        0 /* done */,
                        tile_base + REDUCE_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL);
                    TT_SFPLOAD(
                        p_sfpu::LREG3,
                        MODE,
                        ADDR_MOD_7,
                        0 /* done */,
                        tile_base + REDUCE_FACE_STRIDE + p_sfpu::col_offset::ODD_COL);

                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();
                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG2>();

                    if (ct == 0) {
                        // Seeding from the first tile rather than an identity keeps MAX correct
                        // without needing a format-specific -infinity to start from.
                        TTI_SFPMOV(p_sfpu::LREG0, REDUCE_ACC_REG, 0 /* instr_mod1: plain copy */);
                    } else {
                        reduce_combine<POOL_TYPE, IS_INT, REDUCE_ACC_REG, p_sfpu::LREG0>();
                    }
                }

                reduce_row_fold_columns<POOL_TYPE, IS_INT>();

                if constexpr (POOL_TYPE == PoolType::AVG) {
                    // dest = a * b + c with c = 0.0: scale by the preloaded 1/num_cols.
                    TTI_SFPMUL(
                        REDUCE_ACC_REG,
                        REDUCE_RECIP_REG,
                        p_sfpu::LCONST_0,
                        REDUCE_ACC_REG,
                        0 /* instr_mod1: no negation */);
                }

                // Column 0 of these four tile rows lives in the first tile's left face, even parity.
                TT_SFPSTORE(
                    REDUCE_ACC_REG,
                    MODE,
                    ADDR_MOD_7,
                    0 /* done */,
                    row_base + quad_offset + p_sfpu::col_offset::EVEN_COL);
            }
        }
    }
}

/**
 * @brief Put 1/num_cols into REDUCE_RECIP_REG for the row AVG scaling.
 *
 * A row AVG's divisor is the block's runtime column count, not a compile-time power of two, so
 * unlike the column AVG it cannot ride along as an SFPMULI immediate. The reciprocal is built on
 * the RISC-V side and loaded as two halves, SFPLOADI writing 16 bits at a time.
 *
 * @param num_cols: Columns each row total spans
 * @note Call once per @ref calculate_reduce, before @ref reduce_row_block, which preserves the
 *       register for the whole block.
 */
inline void reduce_row_load_reciprocal(const std::uint32_t num_cols) {
    const float reciprocal = 1.0f / static_cast<float>(num_cols);
    const std::uint32_t bits = __builtin_bit_cast(std::uint32_t, reciprocal);

    TT_SFPLOADI(REDUCE_RECIP_REG, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(REDUCE_RECIP_REG, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Prepare the math thread's SFPU state for a run of reduce calls.
 *
 * The reduce body addresses Dest with quad-relative immediates off the base
 * @ref _llk_math_eltwise_sfpu_start_ programs, so the RWC counters have to start from zero.
 * Everything else the kernel needs - the SFPU config register and ADDR_MOD_7 - is already set up
 * by @ref _llk_math_eltwise_sfpu_init_, and no reduce path claims a programmable constant
 * register or a replay slot, so nothing here can collide with a neighbouring op's init.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 * @tparam IS_FP32_DEST_ACC_EN: Whether Dest holds 32-bit words
 * @note Call after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_reduce.
 */
template <PoolType POOL_TYPE, DataFormat FORMAT, bool IS_FP32_DEST_ACC_EN>
inline void init_reduce() {
    static_assert(
        is_supported_reduce_format(FORMAT), "Unsupported reduce format: expected Float32, Float16_b, Float16 or Int32");
    static_assert(
        POOL_TYPE == PoolType::SUM || POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::MAX,
        "Unsupported pool_type: Quasar PoolType provides SUM, AVG and MAX");

    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Reduce Dest along one axis with the given pooling operator.
 *
 * REDUCE_COL collapses one tile's 32 rows onto its row 0 and is called once per tile, so a
 * multi-tile column reduce can be driven a Dest block at a time. REDUCE_ROW collapses a tile
 * row's columns onto column 0 and is called once for the whole block, because a row's total
 * spans every tile of its tile row.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam REDUCE_DIM: Axis to collapse, values = <REDUCE_COL/REDUCE_ROW>
 * @tparam FORMAT: Math-side data format of the operand
 * @tparam IS_FP32_DEST_ACC_EN: Whether Dest holds 32-bit words
 * @param block_ct_dim: Tiles per tile row; the width a row reduce spans (REDUCE_ROW only)
 * @param block_rt_dim: Tile rows in the block (REDUCE_ROW only)
 * @note Run under VectorMode::RC_custom: this walks Dest itself rather than being driven once
 *       per face, and for REDUCE_ROW it walks past the tile the caller based it on.
 * @note Call @ref init_reduce before this.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIM, DataFormat FORMAT, bool IS_FP32_DEST_ACC_EN>
inline void calculate_reduce(
    [[maybe_unused]] const std::uint32_t block_ct_dim = 1, [[maybe_unused]] const std::uint32_t block_rt_dim = 1) {
    static_assert(
        is_supported_reduce_format(FORMAT), "Unsupported reduce format: expected Float32, Float16_b, Float16 or Int32");
    static_assert(
        REDUCE_DIM == ReduceDim::REDUCE_COL || REDUCE_DIM == ReduceDim::REDUCE_ROW,
        "Unsupported reduce_dim: expected REDUCE_COL or REDUCE_ROW");
    // An integer AVG would have to round its quotient, and the divisor is only a power of two on
    // the column axis. Rounding is a caller-visible choice rather than a detail this kernel should
    // pick, so integer averaging is left to a float reduce plus an explicit typecast.
    static_assert(
        !(POOL_TYPE == PoolType::AVG && reduce_is_int_format<FORMAT>()),
        "Integer AVG reduce is not supported: reduce as a float format and typecast the result");

    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_COL) {
        reduce_col_tile<POOL_TYPE, FORMAT>();
    } else {
        if constexpr (POOL_TYPE == PoolType::AVG) {
            reduce_row_load_reciprocal(block_ct_dim * REDUCE_COLS_PER_TILE);
        }
        reduce_row_block<POOL_TYPE, FORMAT>(block_ct_dim, block_rt_dim);
    }
}

}  // namespace sfpu
}  // namespace ckernel
