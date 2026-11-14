// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// SFPU binary-with-broadcast kernel (BCAST_COL / BCAST_ROW)
// ============================================================================
//
// Tile layout in dest register (SFPU-addressable):
//   A 32x32 tile is 4 faces of 16x16 values, arranged:
//       Face 0 (tile rows  0-15 | tile cols  0-15)
//       Face 1 (tile rows  0-15 | tile cols 16-31)
//       Face 2 (tile rows 16-31 | tile cols  0-15)
//       Face 3 (tile rows 16-31 | tile cols 16-31)
//
//   One SFPLOAD/SFPSTORE moves 4 dest rows x 8 cols (32 lanes). Dest addresses
//   are in units of "4-row x 8-col" slots:
//       addr +0 -> rows 0-3,  cols  0-7 of the face
//       addr +2 -> rows 0-3,  cols  8-15
//       addr +4 -> rows 4-7,  cols  0-7
//       ...
//       addr +14 -> rows 12-15, cols 8-15
//       addr +16 -> start of next face
//
// SFPU lane numbering in a register (32 lanes = 4 sub-rows x 8 columns):
//       lane[i * 8 + c]  i in {0..3}, c in {0..7}
//   After SFPLOAD of rows r..r+3 cols c0..c0+7:
//       LReg[i * 8 + c] = data[r + i][c0 + c]
//
// Broadcast model:
//   - BCAST_COL: column 0 of the bcast tile is replicated across all 32 tile
//                columns, then eltwise-combined with the data tile.
//   - BCAST_ROW: row 0 of the bcast tile is replicated across all 32 tile
//                rows, then eltwise-combined with the data tile.
//
// Implementation strategy (BCAST_COL):
//   For each 4-row band of the source/dest tiles:
//     (1) Load col 0 (8 values, one per SFPU column) into LREG_DATA.
//         After load, only "SFPU column 0" in each 8-lane sub-vector carries
//         the useful value (the data broadcast target); the other 7 SFPU
//         columns hold d[row][cols 1..7] which we must discard.
//     (2) Zero out SFPU cols 1..7 in each 8-lane group with one SFPAND against
//         the precomputed lane mask LREG_MASK (all ones in col 0, zero in cols
//         1..7). A bitwise AND keeps col 0 exactly and cannot turn an Inf or NaN
//         sitting in a discarded column into NaN, which a multiply by a 0/1 mask
//         would, and step (3) would then spread it across the whole row.
//     (3) Broadcast col-0 value to all 8 SFPU columns via 3 stages of
//         "rotate-by-k + add" using SFPSHFT2_MOD1_SUBVEC_SHFLROR1:
//             stage 1: ROR by 1, ADD  -> col {0,1} filled
//             stage 2: ROR by 2, ADD  -> col {0..3} filled
//             stage 3: ROR by 4, ADD  -> col {0..7} filled
//         This works because the masked-out lanes are 0.0, so ADD is
//         equivalent to OR.
//     (4) For each of the 4 target (col-group, face) slots in this row band,
//         load the data tile's corresponding slot, perform the elementwise
//         op against the broadcast value, and store the result.
//
// All "compute" sequences that are identical across row bands are recorded
// into the SFPU replay buffer so the loop body is short.
//
// Implementation strategy (BCAST_ROW):
//   SFPTRANSP + in-place eltwise, with the bcast load hoisted out of the
//   per-band loop. Bcast row 0 lives in face 0 row band 0 and is the same
//   for every output band, so its 4 col-groups are loaded once into
//   LREG4..7 at the start of the full-tile call.
//
//   Per band: load data's 4 col-groups into LREG0..3; SFPTRANSP (acts
//   independently on {LREG0..3} and {LREG4..7}); op LREG_k with LREG4
//   for each k in [0..3], writing ONLY LREG0..3; SFPTRANSP back. Because
//   step 3 leaves LREG4..7 untouched, the second SFPTRANSP is a pure
//   inverse of the first for that group and restores LREG4..7 to their
//   pre-transpose layout, ready for the next band.
//
//   The SUB path uses SFPMAD with LCONST_neg1 (data = bcast*-1 + data)
//   to fuse the negation into the FMA rather than SFPMOV-NEGATE into a
//   scratch LREG, which would corrupt LREG5..7 and force a reload of
//   the bcast every band.
//
// ============================================================================
// Layout constants
// ============================================================================

// One dest-register tile occupies 64 addr units (4 faces x 16 addr/face).
constexpr std::uint32_t DEST_TILE_SIZE_RAW = 64;

// Per-face address offsets (in dest addr units):
//   Face 0 upper-left,  Face 1 upper-right,
//   Face 2 lower-left,  Face 3 lower-right.
constexpr std::uint32_t FACE0_BASE = 0;
constexpr std::uint32_t FACE1_BASE = 16;
constexpr std::uint32_t FACE2_BASE = 32;
constexpr std::uint32_t FACE3_BASE = 48;

// Within a face, each "row band" (4 dest rows) starts every 4 addr units.
constexpr std::uint32_t ROW_BAND_STRIDE = 4;
constexpr std::uint32_t NUM_ROW_BANDS_PER_FACE_HALF = 4;  // 16 rows / 4 rows-per-band
constexpr std::uint32_t ODD_COLS_OFFSET = 2;              // addr +2 selects cols 8..15 of a face

// ============================================================================
// LReg assignments used across the kernel
// ============================================================================
//
// BCAST_COL path:
//   LREG_MASK  (LREG6)  - one-time lane mask: all ones in SFPU col 0, zero elsewhere.
//                         Built once in _sfpu_binary_bcast_init_().
//   LREG_BCAST (LREG0)  - broadcast value (after load+mask+rotate-add).
//   LREG_TMP   (LREG2)  - scratch for rotate-and-add inside the broadcast
//                         sequence; reused as -bcast scratch in the SUB path.
//   LREG_DATA{0..3} (LREG1, LREG3, LREG4, LREG5) - 4 data tile values
//                         pre-loaded per row band so the per-slot binops can
//                         pipeline back-to-back without inter-instruction
//                         NOPs (each binop writes a distinct LREG).
//
// BCAST_ROW path:
//   Uses LREG0..LREG7 transiently during SFPTRANSP phases. LREG6 is NOT
//   preserved as a persistent mask (SFPTRANSP clobbers LREG4..LREG7 window);
//   BCAST_ROW does not need the col-0 lane mask.
//
// BCAST_COL normalize path:
//   LREG0/LREG2 and LREG7/LREG1 carry the two interleaved broadcast chains
//   (mean and inverse standard deviation); LREG1 is LREG_DATA0 and is only
//   loaded once the chains have finished. Replay slots [0, 16) are
//   re-recorded on every call; see the normalize section below.
//
// BCAST_SCALAR normalize path:
//   LREG6/LREG7 retain the negated scalar mean and inverse standard deviation for the
//   tile traversal. This invalidates the BCAST_COL lane mask in LREG6, so a
//   subsequent BCAST_COL operation must call sfpu_bcast_col_init() again.
//   LREG14 (programmable constant 2) is clobbered while extracting each scalar.
//   A subsequent SFPU operation that uses this constant must run its init first.

constexpr std::uint32_t LREG_BCAST = p_sfpu::LREG0;
constexpr std::uint32_t LREG_TMP = p_sfpu::LREG2;
constexpr std::uint32_t LREG_MASK = p_sfpu::LREG6;

// Four distinct LREGs for pipelined per-slot data values in BCAST_COL.
// They must be mutually distinct AND distinct from LREG_BCAST/LREG_TMP/LREG_MASK
// so that the 4 binops can be issued back-to-back without inter-op NOPs.
constexpr std::uint32_t LREG_DATA0 = p_sfpu::LREG1;
constexpr std::uint32_t LREG_DATA1 = p_sfpu::LREG3;
constexpr std::uint32_t LREG_DATA2 = p_sfpu::LREG4;
constexpr std::uint32_t LREG_DATA3 = p_sfpu::LREG5;
constexpr std::uint32_t LREG_SCALAR_NEG_MEAN = p_sfpu::LREG6;
constexpr std::uint32_t LREG_SCALAR_INV_STD = p_sfpu::LREG7;

// SFPSHFT2 Mod1 encoding: rotate right by 1 within each 8-lane sub-vector.
constexpr std::uint32_t SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 3;
constexpr std::uint32_t SFPSHFT_MOD1_ARG_IMM = 1;

// Replay-buffer slots.
constexpr std::uint32_t REPLAY_SLOT_BROADCAST = 0;  // col-0 -> all-cols broadcast
constexpr std::uint32_t REPLAY_LEN_BROADCAST = 11;  // see _record_broadcast_replay_

// BCAST_ROW col-group offsets within a face pair (face N + face N+1), in
// dest addr units. A "col group" is one 4-row x 8-col SFPLOAD/SFPSTORE slot.
// For the upper tile half, faces are 0 (left, cols 0-15) and 1 (right, cols
// 16-31). For each face, +0 selects cols 0-7 and +2 selects cols 8-15.
constexpr std::uint32_t COL_GROUP_OFFSETS[4] = {
    0,                             // face N   cols 0-7
    ODD_COLS_OFFSET,               // face N   cols 8-15
    FACE1_BASE,                    // face N+1 cols 0-7
    FACE1_BASE + ODD_COLS_OFFSET,  // face N+1 cols 8-15
};

// ============================================================================
// Helpers
// ============================================================================

// Build LREG_MASK = {~0, 0, 0, 0, 0, 0, 0, 0} repeated across all 4 sub-rows
// of the register. LReg[15] contains 2*lane_id, so left-shifting it by 28
// leaves zero only for SFPU col 0 in each 8-lane sub-vector; those lanes get
// all ones by predicated SFPLOADI. One SFPAND against it then zeroes cols 1..7
// of a register bitwise, so an Inf or NaN in a discarded column stays out of
// col 0 (a multiply by a 0/1 mask would make it NaN).
inline void _build_lane_mask_col0_() {
    TTI_SFPMOV(0, p_sfpu::LTILEID, LREG_TMP, 0);
    TTI_SFPSHFT(28, 0, LREG_TMP, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, LREG_MASK, 0);
    TTI_SFPSETCC(0, LREG_TMP, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);
    TTI_SFPENCC(0, 0, 0, 0);
}

// Record the mask + stage1 (ROR by 1) + stage2 (ROR by 2) portion of the
// broadcast sequence into the replay buffer. Stage 3 (ROR by 4) is emitted
// inline by _broadcast_stage3_with_data_prefetch_() because keeping replay
// slots short leaves room for other kernels' replay entries if/when
// combined, and because stage 3 interleaves per-slot SFPLOADs into its
// SFPSHFT2 latency slots (data addresses are per-call so can't be recorded).
//
// Sequence recorded (11 SFPU ops):
//    1  SFPAND    LREG_BCAST &= LREG_MASK   (zero SFPU cols 1..7 bitwise)
//    2  SFPSHFT2  LREG_TMP = ROR1(LREG_BCAST)
//    3  SFPNOP
//    4  SFPADD    LREG_BCAST += LREG_TMP
//    5  SFPNOP
//    6  SFPSHFT2  LREG_TMP = ROR1(LREG_BCAST)
//    7  SFPNOP
//    8  SFPSHFT2  LREG_TMP = ROR1(LREG_TMP)
//    9  SFPNOP
//   10  SFPADD    LREG_BCAST += LREG_TMP
//   11  SFPNOP
// No SFPNOP after step 1: SFPAND is a single-cycle op whose result the next
// instruction reads directly (only the SFPMAD-class ops need the slot).
inline void _record_broadcast_replay_() {
    lltt::record(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);

    // (1) Mask: zero out SFPU cols 1..7 bitwise, so an Inf or NaN in a
    //     discarded column cannot reach col 0.
    TTI_SFPAND(0, LREG_MASK, LREG_BCAST, 0);

    // Stage 1: ROR by 1, add.
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;

    // Stage 2: ROR by 2 (ROR1 twice), add.
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
}

// Emit the "rotate by 4, add" tail stage inline (not in the replay buffer),
// because inserting it pushes the recorded length past what we want to
// guarantee fits in a single slot on all SKUs.
//
// The 4 SFPSHFT2 ops form a strict serial chain through LREG_TMP (each is
// 2-cycle and writes the LREG the next op reads), so the 4 inner latency
// slots must be filled. They are filled by interleaving the 4 independent
// per-slot data loads (LREG_DATA0..3 are disjoint from LREG_BCAST/LREG_TMP)
// - this lets us fold the data prefetch into stage 3 for free instead of
// issuing 4 SFPLOADs after the broadcast finishes.
//
// Postcondition: on return, LREG_BCAST holds the fully-broadcast value and
// LREG_DATA0..3 hold the 4 per-slot data values, both ready to be read by
// the next instruction (the trailing SFPNOP drains the final SFPADD's
// 2-cycle latency on LREG_BCAST so callers don't need to insert their own).
inline void _broadcast_stage3_with_data_prefetch_(
    std::uint32_t data_addr0, std::uint32_t data_addr1, std::uint32_t data_addr2, std::uint32_t data_addr3) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, data_addr0);  // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, data_addr1);  // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, data_addr2);  // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, data_addr3);  // fills LREG_TMP latency
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;  // drain SFPADD's 2-cycle latency on LREG_BCAST so callers can read it next cycle
}

// Complete stage 3 when the data LREGs already hold live values. The four
// NOPs cover the dependent rotate chain without clobbering those values.
inline void _broadcast_stage3_preserve_data_() {
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
}

// ============================================================================
// BCAST_COL: per row-band compute kernel
// ============================================================================
//
// For one 4-row band of the tile (identified by the source/dest addresses
// pointing at col 0 of that band):
//   1. Load col-0 values (SFPU col 0 of each 8-lane group carries the useful
//      data; cols 1..7 are noise from adjacent tile cols).
//   2. Mask + broadcast stages 1+2 from the replay buffer, then stage 3
//      (ROR4 + ADD) inline with the 4 per-slot data SFPLOADs interleaved
//      into stage 3's RAW-latency slots (saves 4 inner NOPs per row band).
//   3. Issue the 4 binops back-to-back; each writes a distinct LREG and
//      reads LREG_BCAST, so they pipeline without inter-op NOPs.
//   4. Stream out 4 SFPSTOREs; the first is 4 instructions after the SFPADD
//      that wrote LREG_DATA0, so no leading NOP is needed.
template <BinaryOp BINOP>
inline void _process_col_bcast_row_band_(
    std::uint32_t bcast_col0_addr,
    std::uint32_t left_face_addr,
    std::uint32_t right_face_addr,
    std::uint32_t data_tile_offset,
    std::uint32_t out_tile_offset) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // Slot addresses within this row band (tile-local: face-base + in-face
    // row-band offset + even/odd col-group).
    const std::uint32_t slot0 = left_face_addr;                     // left face, cols 0-7
    const std::uint32_t slot1 = left_face_addr + ODD_COLS_OFFSET;   // left face, cols 8-15
    const std::uint32_t slot2 = right_face_addr;                    // right face, cols 0-7
    const std::uint32_t slot3 = right_face_addr + ODD_COLS_OFFSET;  // right face, cols 8-15

    // (1) Load col 0 values: 4 dest rows x 8 SFPU cols. Only SFPU col 0 in
    //     each 8-lane group holds the column-0 values we want; cols 1..7
    //     hold data at tile-cols 1..7 (we'll zero them out with the mask).
    TT_SFPLOAD(LREG_BCAST, IM, ADDR_MOD_3, bcast_col0_addr);

    // (2) Broadcast: replay records mask+ROR1+ROR2 (stage 1+2). Stage 3
    //     (ROR4+ADD) is emitted inline with the 4 data SFPLOADs interleaved
    //     into the SFPSHFT2 chain's latency slots, so LREG_DATA0..3 are
    //     ready by the time the per-slot binops below start.
    lltt::replay(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);
    _broadcast_stage3_with_data_prefetch_(
        data_tile_offset + slot0, data_tile_offset + slot1, data_tile_offset + slot2, data_tile_offset + slot3);

    // (3) Pipelined per-slot binop. The 4 ops are pairwise independent
    //     (distinct dest LREGs, all reading LREG_BCAST and a distinct
    //     LREG_DATAk), so they issue back-to-back with no NOPs between
    //     them. The helper above already drained the stage-3 SFPADD's
    //     2-cycle latency on LREG_BCAST, and the subsequent 4 SFPSTOREs
    //     provide the trailing 2-cycle drain implicitly - no NOPs needed.
    if constexpr (BINOP == BinaryOp::ADD) {
        TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA0, 0);
        TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA1, 0);
        TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA2, 0);
        TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA3, 0);
    } else if constexpr (BINOP == BinaryOp::SUB) {
        // data - bcast = data + (-bcast). Negate bcast once into LREG_TMP
        // (free after the broadcast sequence) then reuse for all 4 adds.
        TTI_SFPMOV(0, LREG_BCAST, LREG_TMP, 1 /* SFPMOV_MOD1_NEGATE */);
        TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA0, 0);
        TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA1, 0);
        TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA2, 0);
        TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA3, 0);
    } else if constexpr (BINOP == BinaryOp::MUL) {
        TTI_SFPMUL(LREG_DATA0, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA0, 0);
        TTI_SFPMUL(LREG_DATA1, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA1, 0);
        TTI_SFPMUL(LREG_DATA2, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA2, 0);
        TTI_SFPMUL(LREG_DATA3, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA3, 0);
    } else {
        static_assert(
            BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL,
            "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }

    // (5) Streamed stores. The first SFPSTORE issues 4 cycles after the
    //     SFPADD/SFPMUL that wrote LREG_DATA0 - well past the 2-cycle
    //     latency window - so no leading NOP is required.
    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, out_tile_offset + slot0);
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, out_tile_offset + slot1);
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, out_tile_offset + slot2);
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, out_tile_offset + slot3);
}

// Top-level BCAST_COL driver for a single 32x32 tile.
template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_col_full_tile_(
    std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out) {
    const std::uint32_t data_base = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base = dst_index_out * DEST_TILE_SIZE_RAW;

    // Upper tile half (tile rows 0-15): face 0 = left, face 1 = right.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;

        _process_col_bcast_row_band_<BINOP>(
            /* bcast_col0_addr  */ bcast_base + FACE0_BASE + band_off,
            /* left_face_addr   */ FACE0_BASE + band_off,
            /* right_face_addr  */ FACE1_BASE + band_off,
            /* data_tile_offset */ data_base,
            /* out_tile_offset  */ out_base);
    }

    // Lower tile half (tile rows 16-31): face 2 = left, face 3 = right.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;

        _process_col_bcast_row_band_<BINOP>(
            /* bcast_col0_addr  */ bcast_base + FACE2_BASE + band_off,
            /* left_face_addr   */ FACE2_BASE + band_off,
            /* right_face_addr  */ FACE3_BASE + band_off,
            /* data_tile_offset */ data_base,
            /* out_tile_offset  */ out_base);
    }
}

// ============================================================================
// BCAST_COL normalize: out = (data - mean) * inv_std, column 0 of each
// statistics tile broadcast across the tile
// ============================================================================
//
// The two broadcasts are independent (mean: LREG_NORM_MEAN / LREG_NORM_MEAN_TMP,
// inv_std: LREG_NORM_INV_STD / LREG_NORM_INV_STD_TMP, both read LREG_MASK), so
// their mask + rotate-add chains are interleaved A/B/A/B: every dependent
// pair is separated by exactly one instruction of the other chain, which
// covers the 2-cycle SFPU latency without a single SFPNOP. The per-lane
// arithmetic (mask multiply, three rotate-adds, negate, add, multiply) and
// its order are unchanged, so results are bit-identical to the serial form.
//
// Dest addressing is RWC-relative so that the per-band body is a sequence of
// immediates: DEST_TARGET_REG_CFG_MATH_Offset is pointed at the data tile for
// the duration of the call and RWC.Dst walks the 8 row bands (INCRWC 4 per
// band, +16 at the face-pair boundary). The statistics tiles are addressed
// through 10-bit wrapped offsets relative to the data tile (the SFPU dest
// address adder is 10 bits wide, so a negative offset wraps correctly).
//
// The mask + stage 1 + stage 2 + first half of stage 3 of the interleaved
// chains (16 ops, no addresses) are recorded into replay slots [0, 16) once
// per call. This stays inside the SFPU half of the replay buffer (FPU kernels
// own [replay_buf_offset, 32)). The first 12 slots overlap REPLAY_SLOT_BROADCAST:
// the mean chain is exactly the recorded single-chain sequence with its NOP
// slots filled by the inv_std chain (writing LREG7 and LREG1 = LREG_DATA0,
// both dead at that point of the add/sub/mul path), so a subsequent
// BCAST_COL add/sub/mul still works without re-running its init.

constexpr std::uint32_t LREG_NORM_MEAN = p_sfpu::LREG0;      // = LREG_BCAST
constexpr std::uint32_t LREG_NORM_MEAN_TMP = p_sfpu::LREG2;  // = LREG_TMP
constexpr std::uint32_t LREG_NORM_INV_STD = p_sfpu::LREG7;
constexpr std::uint32_t LREG_NORM_INV_STD_TMP = p_sfpu::LREG1;  // = LREG_DATA0, loaded only after the broadcast

constexpr std::uint32_t REPLAY_SLOT_NORM_BROADCAST = 0;
constexpr std::uint32_t REPLAY_LEN_NORM_BROADCAST = 16;
static_assert(
    REPLAY_SLOT_NORM_BROADCAST + REPLAY_LEN_NORM_BROADCAST <= ckernel::math::replay_buf_offset,
    "SFPU replay window must stay below the FPU partition of the replay buffer");

// SFPU dest addresses are 10 bits; offsets relative to the data tile wrap.
constexpr std::uint32_t SFPU_DEST_ADDR_MASK = 0x3FF;

// Band-relative slot immediates: RWC.Dst points at the band's col-0 slot of
// the left face, the right face is +16. The same immediates serve the lower
// half because RWC.Dst is moved to the face 2/3 pair there.
constexpr std::uint32_t NORM_SLOT0 = 0;
constexpr std::uint32_t NORM_SLOT1 = ODD_COLS_OFFSET;
constexpr std::uint32_t NORM_SLOT2 = FACE1_BASE;
constexpr std::uint32_t NORM_SLOT3 = FACE1_BASE + ODD_COLS_OFFSET;

inline std::uint32_t _norm_rel_tile_offset_(std::uint32_t dst_index, std::uint32_t dst_index_data) {
    return ((dst_index - dst_index_data) * DEST_TILE_SIZE_RAW) & SFPU_DEST_ADDR_MASK;
}

// Record the interleaved broadcast block (16 ops). Precondition at replay:
// LREG_NORM_MEAN / LREG_NORM_INV_STD hold the freshly loaded col-0 slots.
//
//    1  SFPMUL    MEAN    *= MASK
//    2  SFPMUL    INV_STD *= MASK
//    3  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 1
//    4  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//    5  SFPADD    MEAN    += MEAN_TMP
//    6  SFPADD    INV_STD += INV_STD_TMP
//    7  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 2
//    8  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//    9  SFPSHFT2  MEAN_TMP    = ROR1(MEAN_TMP)
//   10  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD_TMP)
//   11  SFPADD    MEAN    += MEAN_TMP
//   12  SFPADD    INV_STD += INV_STD_TMP
//   13  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 3, first half
//   14  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//   15  SFPSHFT2  MEAN_TMP    = ROR1(MEAN_TMP)
//   16  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD_TMP)
inline void _record_norm_dual_broadcast_replay_() {
    lltt::record(REPLAY_SLOT_NORM_BROADCAST, REPLAY_LEN_NORM_BROADCAST);

    TTI_SFPMUL(LREG_NORM_MEAN, LREG_MASK, p_sfpu::LCONST_0, LREG_NORM_MEAN, 0);
    TTI_SFPMUL(LREG_NORM_INV_STD, LREG_MASK, p_sfpu::LCONST_0, LREG_NORM_INV_STD, 0);

    TTI_SFPSHFT2(0, LREG_NORM_MEAN, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(LREG_NORM_MEAN, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN, 0);
    TTI_SFPADD(LREG_NORM_INV_STD, p_sfpu::LCONST_1, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD, 0);

    TTI_SFPSHFT2(0, LREG_NORM_MEAN, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(LREG_NORM_MEAN, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN, 0);
    TTI_SFPADD(LREG_NORM_INV_STD, p_sfpu::LCONST_1, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD, 0);

    TTI_SFPSHFT2(0, LREG_NORM_MEAN, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

// Load both col-0 slots of the current band, run the recorded block and the
// second half of stage 3 (two more ROR1 per chain, then the final adds), and
// leave -mean in LREG_NORM_MEAN and inv_std in LREG_NORM_INV_STD, both ready
// to be read by the next instruction.
sfpi_inline void _norm_dual_broadcast_band_(std::uint32_t mean_rel, std::uint32_t inv_std_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(LREG_NORM_MEAN, IM, ADDR_MOD_3, mean_rel);
    TT_SFPLOAD(LREG_NORM_INV_STD, IM, ADDR_MOD_3, inv_std_rel);
    lltt::replay(REPLAY_SLOT_NORM_BROADCAST, REPLAY_LEN_NORM_BROADCAST);
    TTI_SFPSHFT2(0, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(LREG_NORM_MEAN, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN, 0);
    TTI_SFPADD(LREG_NORM_INV_STD, p_sfpu::LCONST_1, LREG_NORM_INV_STD_TMP, LREG_NORM_INV_STD, 0);
    TTI_SFPMOV(0, LREG_NORM_MEAN, LREG_NORM_MEAN, 1 /* SFPMOV_MOD1_NEGATE */);
}

// LREG_DATA0..3 hold the band's data; centre, scale and store them to the
// same slots. The adds / muls / stores each read a value written 4
// instructions earlier, past the 2-cycle latency.
sfpi_inline void _norm_centre_scale_store_band_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA0, 0);
    TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA1, 0);
    TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA2, 0);
    TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA3, 0);
    TTI_SFPMUL(LREG_DATA0, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA0, 0);
    TTI_SFPMUL(LREG_DATA1, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA1, 0);
    TTI_SFPMUL(LREG_DATA2, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA2, 0);
    TTI_SFPMUL(LREG_DATA3, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA3, 0);
    TTI_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, NORM_SLOT0);
    TTI_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, NORM_SLOT1);
    TTI_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, NORM_SLOT2);
    TTI_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, NORM_SLOT3);
}

// Same for a second in-place tile at a wrapped offset from the data tile.
sfpi_inline void _norm_centre_scale_store_band_rel_(std::uint32_t tile_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t slot0 = (tile_rel + NORM_SLOT0) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t slot1 = (tile_rel + NORM_SLOT1) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t slot2 = (tile_rel + NORM_SLOT2) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t slot3 = (tile_rel + NORM_SLOT3) & SFPU_DEST_ADDR_MASK;
    TT_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, slot0);
    TT_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, slot1);
    TT_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, slot2);
    TT_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, slot3);
    TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA0, 0);
    TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA1, 0);
    TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA2, 0);
    TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_NORM_MEAN, LREG_DATA3, 0);
    TTI_SFPMUL(LREG_DATA0, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA0, 0);
    TTI_SFPMUL(LREG_DATA1, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA1, 0);
    TTI_SFPMUL(LREG_DATA2, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA2, 0);
    TTI_SFPMUL(LREG_DATA3, LREG_NORM_INV_STD, p_sfpu::LCONST_0, LREG_DATA3, 0);
    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, slot0);
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, slot1);
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, slot2);
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, slot3);
}

sfpi_inline void _norm_load_data_band_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TTI_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, NORM_SLOT0);
    TTI_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, NORM_SLOT1);
    TTI_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, NORM_SLOT2);
    TTI_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, NORM_SLOT3);
}

// Point the math dest offset at the data tile and record the broadcast block.
// RWC.Dst is 0 on entry (the SFPU wrappers clear it) and ends at 48.
inline void _norm_col_tile_begin_(std::uint32_t dst_index_data) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index_data);
    _record_norm_dual_broadcast_replay_();
}

// Restore the dest offset programmed by _llk_math_eltwise_sfpu_start_ once
// the in-flight stores have drained (config writes must not overtake them).
inline void _norm_col_tile_end_() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
}

// Advance RWC.Dst from the end of the face 0/1 band run (16) to face 2/3 (32).
inline void _norm_col_next_face_pair_() {
    TTI_INCRWC(0, 8, 0, 0);
    TTI_INCRWC(0, 8, 0, 0);
}

inline void _calculate_sfpu_normalize_bcast_col_full_tile_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_mean,
    std::uint32_t /*dst_index_out*/,
    std::uint32_t dst_index_inv_std) {
    const std::uint32_t mean_rel = _norm_rel_tile_offset_(dst_index_mean, dst_index_data);
    const std::uint32_t inv_std_rel = _norm_rel_tile_offset_(dst_index_inv_std, dst_index_data);

    _norm_col_tile_begin_(dst_index_data);
    for (std::uint32_t half = 0; half < 2; half++) {
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
            _norm_dual_broadcast_band_(mean_rel, inv_std_rel);
            _norm_load_data_band_();
            _norm_centre_scale_store_band_();
            TTI_INCRWC(0, ROW_BAND_STRIDE, 0, 0);
        }
        if (half == 0) {
            _norm_col_next_face_pair_();
        }
    }
    _norm_col_tile_end_();
}

inline void _calculate_sfpu_normalize_bcast_col_two_tiles_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_second_data,
    std::uint32_t /*dst_index_out*/,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_inv_std) {
    const std::uint32_t second_rel = _norm_rel_tile_offset_(dst_index_second_data, dst_index_data);
    const std::uint32_t mean_rel = _norm_rel_tile_offset_(dst_index_mean, dst_index_data);
    const std::uint32_t inv_std_rel = _norm_rel_tile_offset_(dst_index_inv_std, dst_index_data);

    _norm_col_tile_begin_(dst_index_data);
    for (std::uint32_t half = 0; half < 2; half++) {
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
            _norm_dual_broadcast_band_(mean_rel, inv_std_rel);
            _norm_load_data_band_();
            _norm_centre_scale_store_band_();
            _norm_centre_scale_store_band_rel_(second_rel);
            TTI_INCRWC(0, ROW_BAND_STRIDE, 0, 0);
        }
        if (half == 0) {
            _norm_col_next_face_pair_();
        }
    }
    _norm_col_tile_end_();
}

inline void _calculate_sfpu_residual_normalize_bcast_col_full_tile_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_residual,
    std::uint32_t /*dst_index_out*/,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_inv_std) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t residual_rel = _norm_rel_tile_offset_(dst_index_residual, dst_index_data);
    const std::uint32_t mean_rel = _norm_rel_tile_offset_(dst_index_mean, dst_index_data);
    const std::uint32_t inv_std_rel = _norm_rel_tile_offset_(dst_index_inv_std, dst_index_data);
    const std::uint32_t res_slot0 = (residual_rel + NORM_SLOT0) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t res_slot1 = (residual_rel + NORM_SLOT1) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t res_slot2 = (residual_rel + NORM_SLOT2) & SFPU_DEST_ADDR_MASK;
    const std::uint32_t res_slot3 = (residual_rel + NORM_SLOT3) & SFPU_DEST_ADDR_MASK;

    _norm_col_tile_begin_(dst_index_data);
    for (std::uint32_t half = 0; half < 2; half++) {
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
            _norm_dual_broadcast_band_(mean_rel, inv_std_rel);
            _norm_load_data_band_();
            // data += residual, one slot at a time through the free scratch
            // LREG (the broadcast temporaries are dead; LREG_NORM_MEAN_TMP is
            // the only LREG not holding data or a statistic).
            TT_SFPLOAD(LREG_NORM_MEAN_TMP, IM, ADDR_MOD_3, res_slot0);
            TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_DATA0, 0);
            TT_SFPLOAD(LREG_NORM_MEAN_TMP, IM, ADDR_MOD_3, res_slot1);
            TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_DATA1, 0);
            TT_SFPLOAD(LREG_NORM_MEAN_TMP, IM, ADDR_MOD_3, res_slot2);
            TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_DATA2, 0);
            TT_SFPLOAD(LREG_NORM_MEAN_TMP, IM, ADDR_MOD_3, res_slot3);
            TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_NORM_MEAN_TMP, LREG_DATA3, 0);
            _norm_centre_scale_store_band_();
            TTI_INCRWC(0, ROW_BAND_STRIDE, 0, 0);
        }
        if (half == 0) {
            _norm_col_next_face_pair_();
        }
    }
    _norm_col_tile_end_();
}

inline void _broadcast_scalar_from_dest_(std::uint32_t scalar_addr) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(LREG_BCAST, IM, ADDR_MOD_3, scalar_addr);
    // The unused statistic-tile lanes can contain infinities after rsqrt.
    // Predicated assignment is required here: multiplying those lanes by a
    // zero mask would produce NaNs rather than zeros.
    // As in _build_lane_mask_col0_, shifting 2*lane_id left by 28 leaves zero
    // only in SFPU column 0.
    TTI_SFPMOV(0, p_sfpu::LTILEID, LREG_TMP, 0);
    TTI_SFPSHFT(28, 0, LREG_TMP, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPSETCC(0, LREG_TMP, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(LREG_BCAST, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    // SFPCONFIG vertically broadcasts the first eight lanes. Moving the
    // result back to LREG0 makes scalar lane zero available in every 8-lane
    // sub-vector; the inline rotate-and-add sequence fills each one.
    // ttsim v1.10.5 permits non-uniform source lanes only for LREG14.
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG14, LREG_BCAST, 0);
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
    _broadcast_stage3_preserve_data_();
}

inline void _process_scalar_normalize_row_band_(
    std::uint32_t left_face_addr,
    std::uint32_t right_face_addr,
    std::uint32_t data_tile_offset,
    std::uint32_t out_tile_offset) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t slot0 = left_face_addr;
    const std::uint32_t slot1 = left_face_addr + ODD_COLS_OFFSET;
    const std::uint32_t slot2 = right_face_addr;
    const std::uint32_t slot3 = right_face_addr + ODD_COLS_OFFSET;

    TT_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, data_tile_offset + slot0);
    TT_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, data_tile_offset + slot1);
    TT_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, data_tile_offset + slot2);
    TT_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, data_tile_offset + slot3);

    TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA0, 0);
    TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA1, 0);
    TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA2, 0);
    TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA3, 0);

    TTI_SFPMUL(LREG_DATA0, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA0, 0);
    TTI_SFPMUL(LREG_DATA1, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA1, 0);
    TTI_SFPMUL(LREG_DATA2, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA2, 0);
    TTI_SFPMUL(LREG_DATA3, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA3, 0);

    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, out_tile_offset + slot0);
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, out_tile_offset + slot1);
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, out_tile_offset + slot2);
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, out_tile_offset + slot3);
}

inline void _calculate_sfpu_normalize_bcast_scalar_full_tile_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_out,
    std::uint32_t dst_index_inv_std) {
    const std::uint32_t data_base = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t mean_addr = dst_index_mean * DEST_TILE_SIZE_RAW;
    const std::uint32_t inv_std_addr = dst_index_inv_std * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base = dst_index_out * DEST_TILE_SIZE_RAW;

    // The statistics are scalar and invariant across all eight row bands.
    // Broadcast each once and retain -mean/inv_std in LREG6/LREG7 for the
    // complete tile traversal. LREG6 deliberately borrows the BCAST_COL mask.
    _broadcast_scalar_from_dest_(mean_addr);
    TTI_SFPMOV(0, LREG_BCAST, LREG_SCALAR_NEG_MEAN, 1 /* SFPMOV_MOD1_NEGATE */);
    _broadcast_scalar_from_dest_(inv_std_addr);
    TTI_SFPMOV(0, LREG_BCAST, LREG_SCALAR_INV_STD, 0);

    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;
        _process_scalar_normalize_row_band_(FACE0_BASE + band_off, FACE1_BASE + band_off, data_base, out_base);
    }
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;
        _process_scalar_normalize_row_band_(FACE2_BASE + band_off, FACE3_BASE + band_off, data_base, out_base);
    }
}

// ============================================================================
// BCAST_ROW: driver (SFPTRANSP + in-place)
// ============================================================================
//
// Setup (once per full-tile call, hoisted out of the per-band loop):
//   0. Load bcast tile's 4 col-groups (face 0, row band 0) into LREG4..7.
//      Bcast row 0 lives entirely in face 0 (upper-left), row band 0.
//
// Per band (8 bands per tile):
//   1. Load data's 4 col-groups into LREG0..3.
//   2. SFPTRANSP (transposes {LREG0..3} and {LREG4..7} independently).
//      Now:
//        LREG_k   (k in 0..3) = data row (band_base + k), 32 cols across sub-rows.
//        LREG_{4+k}           = bcast row k, 32 cols across sub-rows.
//        Only LREG4 (bcast row 0) is useful; LREG5..7 are the transposed
//        junk of bcast rows 1..3 and are intentionally left untouched so
//        that the second SFPTRANSP can undo this one for the 4..7 group.
//   3. For each k in 0..3: LREG_k = LREG_k OP LREG4  (data row op bcast row 0).
//      Critically, this block must NOT write LREG4..7 - see SUB note below.
//   4. SFPTRANSP back. Since step 3 left LREG4..7 unchanged, the second
//      SFPTRANSP is a pure inverse of the first on that group, so LREG4..7
//      return to the face-major layout loaded in step 0. The hoisted
//      bcast therefore survives verbatim into the next band.
//   5. Store LREG0..3 to the 4 col-groups of the output band.

template <BinaryOp BINOP>
inline void _process_row_bcast_data_band_(
    std::uint32_t data_tile_base, std::uint32_t out_tile_base, std::uint32_t face_pair_base, std::uint32_t band_off) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_slot_base = data_tile_base + face_pair_base + band_off;
    const std::uint32_t out_slot_base = out_tile_base + face_pair_base + band_off;

    // (1) Load data's 4 col-groups into LREG0..3. Bcast (LREG4..7) was
    //     pre-loaded once by the caller and survives across iterations.
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[3]);

    // (2) Transpose: LREG0..3 become the 4 data rows; LREG4..7 become the
    //     transposed bcast rows (of which only LREG4 = bcast row 0 is useful).
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (3) In-register eltwise: LREG_k = LREG_k OP LREG4 for k in 0..3.
    //
    // Pipelining: the 4 binops all read LREG4 and a distinct LREG_k, and each
    // writes a distinct LREG_k. They are pairwise independent, so we can
    // issue them back-to-back with no NOPs between. Only a single trailing
    // NOP is needed before SFPTRANSP reads LREG0..3 (2-cycle SFPADD/SFPMUL
    // latency).
    //
    // IMPORTANT: this block must not write LREG4..7. The persistent bcast
    // values rely on the second SFPTRANSP being a pure inverse of the first
    // for that group. The SUB path uses SFPMAD with LCONST_neg1
    // (data = bcast * -1 + data) to fuse the negation into the FMA instead
    // of the more obvious SFPMOV-NEGATE-into-LREG5 + SFPADD pattern, which
    // would corrupt LREG5 and force a reload of the bcast every band.
    if constexpr (BINOP == BinaryOp::ADD) {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    } else if constexpr (BINOP == BinaryOp::SUB) {
        // data - bcast = bcast * (-1) + data, fused into a single SFPMAD per
        // element so we never write LREG4..7 and the bcast survives the
        // second SFPTRANSP.
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
    } else if constexpr (BINOP == BinaryOp::MUL) {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    } else {
        static_assert(
            BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL,
            "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }
    TTI_SFPNOP;  // single drain before SFPTRANSP reads LREG0..3.

    // (4) Transpose back. Since step (3) left LREG4..7 untouched, this also
    //     restores the bcast registers to their pre-transpose face-major
    //     layout, ready for the next band's first SFPTRANSP.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (5) Store LREG0..3 to the 4 col-groups of the output band.
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[1]);
    TT_SFPSTORE(p_sfpu::LREG2, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[2]);
    TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[3]);
}

template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_row_full_tile_(
    std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_base = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base = dst_index_out * DEST_TILE_SIZE_RAW;

    // Hoisted bcast load: bcast row 0 lives in face 0 row band 0 and is
    // identical for every output band, so we load its 4 col-groups exactly
    // once. The per-band binop is structured to leave LREG4..7 unchanged,
    // so the second SFPTRANSP per band restores them to this face-major
    // layout.
    const std::uint32_t bcast_row0_base = bcast_base + FACE0_BASE;
    TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[3]);

    // Upper half (face 0/1): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        _process_row_bcast_data_band_<BINOP>(data_base, out_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }

    // Lower half (face 2/3): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        _process_row_bcast_data_band_<BINOP>(data_base, out_base, FACE2_BASE, band * ROW_BAND_STRIDE);
    }
}

// ============================================================================
// Public API
// ============================================================================

template <BinaryOp BINOP, BroadcastType Dim>
inline void _calculate_sfpu_binary_bcast_full_tile_(
    std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out) {
    static_assert(
        Dim == BroadcastType::COL || Dim == BroadcastType::ROW,
        "SFPU binary bcast only supports BroadcastType::COL / BroadcastType::ROW");
    if constexpr (Dim == BroadcastType::COL) {
        _calculate_sfpu_binary_bcast_col_full_tile_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
    } else {
        _calculate_sfpu_binary_bcast_row_full_tile_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
    }
}

template <BroadcastType Dim>
inline void _sfpu_binary_bcast_init_() {
    static_assert(
        Dim == BroadcastType::COL || Dim == BroadcastType::ROW,
        "SFPU binary bcast only supports BroadcastType::COL / BroadcastType::ROW");
    // Initialize SFPU config register (matches sibling SFPU init helpers,
    // e.g. _init_add_top_row_, init_reduce_*). Required before any replay
    // recording or SFPCONFIG-based lane-mask setup below.
    _init_sfpu_config_reg();

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (Dim == BroadcastType::COL) {
        // Build persistent col-0 lane mask in LREG_MASK.
        _build_lane_mask_col0_();

        // Record the broadcast helper (mask + ROR1 + ROR2) into replay slot 0.
        _record_broadcast_replay_();
    }
    // BCAST_ROW: no persistent state needed - the kernel operates purely
    // on per-tile LREG contents via SFPTRANSP + SFPLOAD/SFPSTORE.
}

}  // namespace sfpu
}  // namespace ckernel
