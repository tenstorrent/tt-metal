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

constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_ROTATE_COLS = 3;           // SFPU col X -> X+1, col 7 wraps to col 0
constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL = 4;  // SFPU col X -> X+1, col 0 <- 0
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
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, 0 /*instr_mod1*/);  // dst = dst + bcast
    } else if constexpr (BINOP == BinaryOp::SUB) {
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, BCAST_MAD_MOD1_NEGATE_C);  // dst = dst - bcast
    } else {
        TTI_SFPMUL(DST, BCAST, p_sfpu::LCONST_0, DST, 0 /*instr_mod1*/);  // dst = dst * bcast
    }
}

// Bit mask isolating SFPU col 0: all ones in col 0, all zeros in cols 1..7 of every row. Col 0 is
// isolated and folded bitwise, never arithmetically, so Inf/NaN in the ignored columns cannot leak
// into the broadcast (0 * Inf = NaN) and a -0.0 in col 0 keeps its sign.
inline void binary_bcast_build_col0_mask_() {
    TTI_SFPLOADI(BCAST_COL_LREG_TMP, sfpi::SFPLOADI_MOD0_SHORT, 0xFFFF);  // all lanes 0xFFFFFFFF
    // {0,~0,~0,~0,~0,~0,~0,~0} per row
    TTI_SFPSHFT2(0 /*imm12*/, BCAST_COL_LREG_TMP, BCAST_COL_LREG_MASK, BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);  // 2-cycle global shift
    TTI_SFPNOT(BCAST_COL_LREG_MASK, BCAST_COL_LREG_MASK);                 // {~0,0,0,0,0,0,0,0}
}

// Replay slots: COL and ROW never run in the same call, so both start at slot 0
constexpr std::uint32_t BCAST_REPLAY_DEPTH = 32;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_LEN = 9;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_SLOT = BCAST_COL_FOLD_REPLAY_SLOT + BCAST_COL_FOLD_REPLAY_LEN;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_LEN = 5;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_LEN = 6;
// Replay slots [0, *_SLOTS_USED) of bank 0 that init records; the init_binary_bcast doc quotes these
constexpr std::uint32_t BCAST_COL_REPLAY_SLOTS_USED = BCAST_COL_OP_REPLAY_SLOT + BCAST_COL_OP_REPLAY_LEN;
constexpr std::uint32_t BCAST_ROW_REPLAY_SLOTS_USED = BCAST_ROW_OP_REPLAY_SLOT + BCAST_ROW_OP_REPLAY_LEN;
static_assert(BCAST_COL_REPLAY_SLOTS_USED <= BCAST_REPLAY_DEPTH, "COL replay bodies must fit");
static_assert(BCAST_ROW_REPLAY_SLOTS_USED <= BCAST_REPLAY_DEPTH, "ROW replay body must fit");
static_assert(
    BCAST_COL_REPLAY_SLOTS_USED == 14 && BCAST_ROW_REPLAY_SLOTS_USED == 6,
    "update the replay-slot ranges in the init_binary_bcast doc");

// Issue LEN instructions recorded at SLOT of the math thread's replay buffer
template <std::uint32_t SLOT, std::uint32_t LEN>
inline void binary_bcast_replay_() {
    TTI_REPLAY(SLOT, LEN, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// Mask + fold stages 1 and 2 (distances 1 and 2) of the col-0 broadcast. The other lanes are bit
// zero after the AND, so each OR copies col 0 bit-exactly into the lanes the rotate reached.
inline void binary_bcast_col_fold_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t M = BCAST_COL_LREG_MASK;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    TTI_SFPAND(M, B);  // clear SFPU cols 1..7

    // Stage 1: fold distance 1
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPOR(T, B);

    // Stage 2: fold distance 2
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPOR(T, B);
}

// Last fold OR of stage 3, then the binop on all four data slots
template <BinaryOp BINOP>
inline void binary_bcast_col_op_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;

    TTI_SFPOR(BCAST_COL_LREG_TMP, B);  // B = bcast[row][0] in all 8 cols

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
    TT_SFPLOAD(B, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    // binary_bcast_col_fold_
    binary_bcast_replay_<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>();

    // Stage 3: fold distance 4; the data loads fill each shift's latency slot
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TT_SFPLOAD(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
    // binary_bcast_col_op_
    binary_bcast_replay_<BCAST_COL_OP_REPLAY_SLOT, BCAST_COL_OP_REPLAY_LEN>();

    TT_SFPSTORE(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPSTORE(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

// Transpose to row order, binop against hoisted bcast row 0, transpose back
template <BinaryOp BINOP>
inline void binary_bcast_row_op_() {
    constexpr std::uint32_t D = BCAST_ROW_LREG_DATA;
    constexpr std::uint32_t B = BCAST_ROW_LREG_BCAST;

    // LREG0-3 now hold one tile row each; LREG4 holds bcast row 0
    TTI_SFPTRANSP;

    // Never write LREG4-7 so the next transpose restores the hoisted bcast rows
    binary_bcast_op_<BINOP, D + 0 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 1 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 2 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 3 /*DST*/, B>();

    TTI_SFPTRANSP;  // back to store order
}

// Bcast rows 0-3 into LREG4-7, hoisted for the whole tile
inline void binary_bcast_row_hoist_(const std::uint32_t bcast_addr) {
    constexpr std::uint32_t B0 = BCAST_ROW_LREG_BCAST + 0;
    constexpr std::uint32_t B1 = BCAST_ROW_LREG_BCAST + 1;
    constexpr std::uint32_t B2 = BCAST_ROW_LREG_BCAST + 2;
    constexpr std::uint32_t B3 = BCAST_ROW_LREG_BCAST + 3;

    TT_SFPLOAD(B0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(B1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPLOAD(B2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(B3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

inline void binary_bcast_row_band_(const std::uint32_t data_addr, const std::uint32_t out_addr) {
    constexpr std::uint32_t D0 = BCAST_ROW_LREG_DATA + 0;
    constexpr std::uint32_t D1 = BCAST_ROW_LREG_DATA + 1;
    constexpr std::uint32_t D2 = BCAST_ROW_LREG_DATA + 2;
    constexpr std::uint32_t D3 = BCAST_ROW_LREG_DATA + 3;

    TT_SFPLOAD(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPLOAD(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);

    // binary_bcast_row_op_
    binary_bcast_replay_<BCAST_ROW_OP_REPLAY_SLOT, BCAST_ROW_OP_REPLAY_LEN>();

    TT_SFPSTORE(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPSTORE(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

/**
 * @brief Init step of the broadcast binary op: records the per-band replay bodies once.
 *
 * COL also builds the col-0 bit mask in LREG6 and records the op-independent mask + fold body
 * alongside the BINOP tail; ROW records its transpose + BINOP body. Every recorded instruction is an
 * immediate, so @ref calculate_binary_bcast only replays them. The address mode and SFPU config the
 * walk rides on are set by @ref _llk_math_eltwise_sfpu_init_.
 *
 * @tparam BINOP: Binary operation, values = <ADD/SUB/MUL> (float operands only)
 * @tparam BCAST_DIM: Broadcast axis, values = <COL/ROW>
 * @note Call @ref _llk_math_eltwise_sfpu_init_ before this, and @ref calculate_binary_bcast with the
 *       same BINOP and BCAST_DIM after it.
 * @note Re-run this before resuming binary_bcast after any op that records into the math thread's
 *       replay bank 0 slots it uses - 0..13 for COL (BCAST_COL_REPLAY_SLOTS_USED), 0..5 for ROW
 *       (BCAST_ROW_REPLAY_SLOTS_USED) - or, for COL, writes LREG6 (a ROW binary_bcast call does both).
 */
template <BinaryOp BINOP, BroadcastType BCAST_DIM>
inline void init_binary_bcast() {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");

    if constexpr (BCAST_DIM == BroadcastType::COL) {
        binary_bcast_build_col0_mask_();
        load_replay_buf<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>([] { binary_bcast_col_fold_(); });
        load_replay_buf<BCAST_COL_OP_REPLAY_SLOT, BCAST_COL_OP_REPLAY_LEN>([] { binary_bcast_col_op_<BINOP>(); });
    } else {
        load_replay_buf<BCAST_ROW_OP_REPLAY_SLOT, BCAST_ROW_OP_REPLAY_LEN>([] { binary_bcast_row_op_<BINOP>(); });
    }
}

/**
 * @brief Elementwise binary op of one whole 32x32 Dest tile against a broadcast row or column.
 *
 * COL replicates column 0 of the bcast tile across all 32 columns, ROW replicates its row 0 down all
 * 32 rows: `out[r][c] = data[r][c] OP bcast[r][0]` or `data[r][c] OP bcast[0][c]`. The tile is walked
 * as four-row bands over the two face pairs; the per-band body is issued from the replay buffer.
 * COL isolates column 0 with a bit mask and folds it across the 8 SFPU columns with rotate-and-OR,
 * so only column 0 is ever read as a value (Inf/NaN elsewhere are ignored, -0.0 is kept); ROW transposes each band
 * against a bcast row hoisted into LREG4-7 for the whole tile. The output tile may alias either input.
 *
 * @tparam BINOP: Binary operation, values = <ADD/SUB/MUL> (float operands only)
 * @tparam BCAST_DIM: Broadcast axis, values = <COL/ROW>
 * @param dst_index_data: Dest tile index of the elementwise operand.
 * @param dst_index_bcast: Dest tile index of the tile supplying the broadcast row or column.
 * @param dst_index_out: Dest tile index the result is written to.
 * @note Run this once per tile under VectorMode::None, not once per face - the walk covers the whole
 *       tile from section-relative addresses.
 * @note Overwrites LREG0-5 (COL; reads the LREG6 mask) or LREG0-7 (ROW), and replays the bodies
 *       @ref init_binary_bcast recorded from slot 0 of the math thread's replay buffer.
 * @note Call @ref init_binary_bcast with the same BINOP and BCAST_DIM before this.
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
        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_col_band_(
                    bcast_base + offset /*bcast_addr*/,
                    data_base + offset /*data_addr*/,
                    out_base + offset /*out_addr*/);
            }
        }
    } else {
        binary_bcast_row_hoist_(bcast_base);

        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_row_band_(data_base + offset /*data_addr*/, out_base + offset /*out_addr*/);
            }
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
