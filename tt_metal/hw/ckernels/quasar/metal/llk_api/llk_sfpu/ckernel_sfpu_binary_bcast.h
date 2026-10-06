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

namespace ckernel {
namespace sfpu {

// Dest geometry of a 32x32 tile: 64 addr units, one per 16-datum face row, face f at 16f. One
// SFPLOAD covers four face rows x 8 datums, address bit 1 picking the even or odd columns.
constexpr std::uint32_t BCAST_DEST_TILE_STRIDE = 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
constexpr std::uint32_t BCAST_FACE_STRIDE = FACE_R_DIM;           // face 0 -> face 1
constexpr std::uint32_t BCAST_FACE_PAIR_STRIDE = 2 * FACE_R_DIM;  // faces 0/1 -> faces 2/3
constexpr std::uint32_t BCAST_ROW_BAND_STRIDE = 4;                // face rows per SFPLOAD
constexpr std::uint32_t BCAST_BANDS_PER_FACE_PAIR = FACE_R_DIM / BCAST_ROW_BAND_STRIDE;
constexpr std::uint32_t BCAST_FACE_PAIRS = 2;

// The four slots of a 4-row band: both column parities of the left face, then of the right face.
constexpr std::uint32_t BCAST_LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t BCAST_LEFT_ODD = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t BCAST_RIGHT_EVEN = BCAST_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t BCAST_RIGHT_ODD = BCAST_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

constexpr std::uint32_t BCAST_SFPLOADI_MOD0_FP16B = 0x0;               // SFPLOADI immediate is bfloat16
constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_ROTATE_COLS = 3;           // SFPU col X -> X+1, col 7 wraps to col 0
constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL = 4;  // SFPU col X -> X+1, col 0 <- 0
constexpr std::uint32_t BCAST_MAD_MOD1_NEGATE_A = 0x1;                 // SFPMAD/SFPADD: negate src_a
constexpr std::uint32_t BCAST_MAD_MOD1_NEGATE_C = 0x2;                 // SFPMAD/SFPADD: negate src_c

// COL register plan
constexpr std::uint32_t BCAST_COL_LREG_BCAST = p_sfpu::LREG0;
constexpr std::uint32_t BCAST_COL_LREG_TMP = p_sfpu::LREG2;
constexpr std::uint32_t BCAST_COL_LREG_MASK = p_sfpu::LREG6;
constexpr std::uint32_t BCAST_COL_LREG_DATA0 = p_sfpu::LREG1;
constexpr std::uint32_t BCAST_COL_LREG_DATA1 = p_sfpu::LREG3;
constexpr std::uint32_t BCAST_COL_LREG_DATA2 = p_sfpu::LREG4;
constexpr std::uint32_t BCAST_COL_LREG_DATA3 = p_sfpu::LREG5;

// ROW register plan: data in LREG0-3, hoisted bcast rows in LREG4-7 (only LREG4 read post-transpose)
constexpr std::uint32_t BCAST_ROW_LREG_DATA = p_sfpu::LREG0;
constexpr std::uint32_t BCAST_ROW_LREG_BCAST = p_sfpu::LREG4;

template <BinaryOp BINOP, std::uint32_t DST, std::uint32_t BCAST>
inline void binary_bcast_op_() {
    static_assert(
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL,
        "binary_bcast only supports ADD, SUB and MUL");

    if constexpr (BINOP == BinaryOp::ADD) {
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, 0 /* instr_mod1 */);  // dst = dst + bcast
    } else if constexpr (BINOP == BinaryOp::SUB) {
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, BCAST_MAD_MOD1_NEGATE_C);  // dst = dst - bcast
    } else {
        TTI_SFPMUL(DST, BCAST, p_sfpu::LCONST_0, DST, 0 /* instr_mod1 */);  // dst = dst * bcast
    }
}

inline void binary_bcast_build_col0_mask_() {
    TTI_SFPLOADI(BCAST_COL_LREG_TMP, BCAST_SFPLOADI_MOD0_FP16B, p_sfpu::kCONST_1_FP16B);  // all lanes 1.0
    // {0,1,1,1,1,1,1,1} per row
    TTI_SFPSHFT2(0 /* imm12 */, BCAST_COL_LREG_TMP, BCAST_COL_LREG_MASK, BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);  // 2-cycle global shift
    // mask = 1 - mask = {1,0,0,0,0,0,0,0}
    TTI_SFPMAD(BCAST_COL_LREG_MASK, p_sfpu::LCONST_1, p_sfpu::LCONST_1, BCAST_COL_LREG_MASK, BCAST_MAD_MOD1_NEGATE_A);
}

// Replay slots: COL and ROW never run in the same call, so both start at slot 0
constexpr std::uint32_t BCAST_REPLAY_DEPTH = 32;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_LEN = 12;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_SLOT = BCAST_COL_FOLD_REPLAY_SLOT + BCAST_COL_FOLD_REPLAY_LEN;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_LEN = 5;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_LEN = 6;
static_assert(BCAST_COL_OP_REPLAY_SLOT + BCAST_COL_OP_REPLAY_LEN <= BCAST_REPLAY_DEPTH, "COL replay bodies must fit");
static_assert(BCAST_ROW_OP_REPLAY_SLOT + BCAST_ROW_OP_REPLAY_LEN <= BCAST_REPLAY_DEPTH, "ROW replay body must fit");

// Mask + fold stages 1 and 2 (distances 1 and 2) of the col-0 broadcast
inline void binary_bcast_col_fold_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t M = BCAST_COL_LREG_MASK;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    TTI_SFPMUL(B, M, p_sfpu::LCONST_0, B, 0 /* instr_mod1 */);  // zero SFPU cols 1..7
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

    // Stage 1: fold distance 1
    TTI_SFPSHFT2(0 /* imm12 */, B, T, ROT);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPADD(p_sfpu::LCONST_1, B, T, B, 0 /* instr_mod1 */);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

    // Stage 2: fold distance 2
    TTI_SFPSHFT2(0 /* imm12 */, B, T, ROT);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPSHFT2(0 /* imm12 */, T, T, ROT);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPADD(p_sfpu::LCONST_1, B, T, B, 0 /* instr_mod1 */);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
}

// Last fold add of stage 3, then the binop on all four data slots
template <BinaryOp BINOP>
inline void binary_bcast_col_op_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;

    TTI_SFPADD(p_sfpu::LCONST_1, B, BCAST_COL_LREG_TMP, B, 0 /* instr_mod1 */);  // B = bcast[row][0] in all 8 cols

    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA0, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA1, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA2, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA3, B>();
}

inline void binary_bcast_col_band_(
    const std::uint32_t bcast_addr, const std::uint32_t data_addr, const std::uint32_t out_addr) {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t D0 = BCAST_COL_LREG_DATA0;
    constexpr std::uint32_t D1 = BCAST_COL_LREG_DATA1;
    constexpr std::uint32_t D2 = BCAST_COL_LREG_DATA2;
    constexpr std::uint32_t D3 = BCAST_COL_LREG_DATA3;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    // SFPU col 0 = tile col 0 of the band's 4 rows
    TT_SFPLOAD(B, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, bcast_addr + BCAST_LEFT_EVEN);
    // binary_bcast_col_fold_
    TTI_REPLAY(
        BCAST_COL_FOLD_REPLAY_SLOT,
        BCAST_COL_FOLD_REPLAY_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    // Stage 3: fold distance 4; the data loads fill each shift's latency slot
    TTI_SFPSHFT2(0 /* imm12 */, B, T, ROT);
    TT_SFPLOAD(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_LEFT_EVEN);
    TTI_SFPSHFT2(0 /* imm12 */, T, T, ROT);
    TT_SFPLOAD(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_LEFT_ODD);
    TTI_SFPSHFT2(0 /* imm12 */, T, T, ROT);
    TT_SFPLOAD(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_RIGHT_EVEN);
    TTI_SFPSHFT2(0 /* imm12 */, T, T, ROT);
    TT_SFPLOAD(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_RIGHT_ODD);
    // binary_bcast_col_op_
    TTI_REPLAY(
        BCAST_COL_OP_REPLAY_SLOT,
        BCAST_COL_OP_REPLAY_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    TT_SFPSTORE(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_LEFT_EVEN);
    TT_SFPSTORE(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_LEFT_ODD);
    TT_SFPSTORE(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_RIGHT_EVEN);
    TT_SFPSTORE(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_RIGHT_ODD);
}

// Transpose to row order, binop against hoisted bcast row 0, transpose back
template <BinaryOp BINOP>
inline void binary_bcast_row_op_() {
    constexpr std::uint32_t D = BCAST_ROW_LREG_DATA;
    constexpr std::uint32_t B = BCAST_ROW_LREG_BCAST;

    // LREG0-3 now hold one tile row each; LREG4 holds bcast row 0
    TTI_SFPTRANSP;

    // Never write LREG4-7 so the next transpose restores the hoisted bcast rows
    binary_bcast_op_<BINOP, D + 0, B>();
    binary_bcast_op_<BINOP, D + 1, B>();
    binary_bcast_op_<BINOP, D + 2, B>();
    binary_bcast_op_<BINOP, D + 3, B>();

    TTI_SFPTRANSP;  // back to store order
}

inline void binary_bcast_row_band_(const std::uint32_t data_addr, const std::uint32_t out_addr) {
    constexpr std::uint32_t D = BCAST_ROW_LREG_DATA;

    TT_SFPLOAD(D + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_LEFT_EVEN);
    TT_SFPLOAD(D + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_LEFT_ODD);
    TT_SFPLOAD(D + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_RIGHT_EVEN);
    TT_SFPLOAD(D + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, data_addr + BCAST_RIGHT_ODD);

    // binary_bcast_row_op_
    TTI_REPLAY(
        BCAST_ROW_OP_REPLAY_SLOT,
        BCAST_ROW_OP_REPLAY_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    TT_SFPSTORE(D + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_LEFT_EVEN);
    TT_SFPSTORE(D + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_LEFT_ODD);
    TT_SFPSTORE(D + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_RIGHT_EVEN);
    TT_SFPSTORE(D + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, out_addr + BCAST_RIGHT_ODD);
}

/**
 * @brief Init step of the broadcast binary op; validates the broadcast axis only.
 *
 * Nothing is programmed here: the address mode and SFPU config the walk rides on are already set by
 * @ref _llk_math_eltwise_sfpu_init_, and the replay bodies depend on the binary op, so
 * @ref calculate_binary_bcast records them itself.
 *
 * @tparam BCAST_DIM: Broadcast axis, values = <COL/ROW>
 * @note Call @ref _llk_math_eltwise_sfpu_init_ before this, and @ref calculate_binary_bcast with the
 *       same BCAST_DIM after it.
 */
template <BroadcastType BCAST_DIM>
inline void init_binary_bcast() {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");
}

/**
 * @brief Elementwise binary op of one whole 32x32 Dest tile against a broadcast row or column.
 *
 * COL replicates column 0 of the bcast tile across all 32 columns, ROW replicates its row 0 down all
 * 32 rows: `out[r][c] = data[r][c] OP bcast[r][0]` or `data[r][c] OP bcast[0][c]`. The tile is walked
 * as four-row bands over the two face pairs; the per-band body is issued from the replay buffer.
 * COL folds column 0 across the 8 SFPU columns with masked rotate-and-add, ROW transposes each band
 * against a bcast row hoisted into LREG4-7 for the whole tile. The output tile may alias either input.
 *
 * @tparam BINOP: Binary operation, values = <ADD/SUB/MUL> (float operands only)
 * @tparam BCAST_DIM: Broadcast axis, values = <COL/ROW>
 * @param dst_index_data: Dest tile index of the elementwise operand.
 * @param dst_index_bcast: Dest tile index of the tile supplying the broadcast row or column.
 * @param dst_index_out: Dest tile index the result is written to.
 * @note Run this once per tile under VectorMode::None, not once per face - the walk covers the whole
 *       tile from section-relative addresses.
 * @note Overwrites LREG0-6 (COL) or LREG0-7 (ROW), and re-records the replay buffer from slot 0 on
 *       the math thread.
 * @note Call @ref init_binary_bcast with the same BCAST_DIM before this.
 */
template <BinaryOp BINOP, BroadcastType BCAST_DIM>
inline void calculate_binary_bcast(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_bcast, const std::uint32_t dst_index_out) {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");

    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t bcast_base = dst_index_bcast * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;

    if constexpr (BCAST_DIM == BroadcastType::COL) {
        binary_bcast_build_col0_mask_();
        load_replay_buf<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>([] { binary_bcast_col_fold_(); });
        load_replay_buf<BCAST_COL_OP_REPLAY_SLOT, BCAST_COL_OP_REPLAY_LEN>([] { binary_bcast_col_op_<BINOP>(); });

        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_col_band_(bcast_base + offset, data_base + offset, out_base + offset);
            }
        }
    } else {
        // Bcast rows 0-3, hoisted for the whole tile
        TT_SFPLOAD(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, bcast_base + BCAST_LEFT_EVEN);
        TT_SFPLOAD(p_sfpu::LREG5, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, bcast_base + BCAST_LEFT_ODD);
        TT_SFPLOAD(p_sfpu::LREG6, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, bcast_base + BCAST_RIGHT_EVEN);
        TT_SFPLOAD(p_sfpu::LREG7, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, bcast_base + BCAST_RIGHT_ODD);
        load_replay_buf<BCAST_ROW_OP_REPLAY_SLOT, BCAST_ROW_OP_REPLAY_LEN>([] { binary_bcast_row_op_<BINOP>(); });

        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_row_band_(data_base + offset, out_base + offset);
            }
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
