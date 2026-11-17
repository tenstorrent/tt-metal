// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "cmath_common.h"
#include "llk_assert.h"
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
//   BCAST_ROW does not need the col-0 lane mask. The band body is recorded
//   into SFPU replay slots 0..15 per full-tile call, and (without
//   MM_THROTTLE) ADDR_MOD_6 is reprogrammed per call; that overwrites what
//   another SFPU op set at init (COL bcast, reduce, fast exp, abs, i0, ...),
//   which must re-run its init after a ROW call. The
//   call leaves the dest base register pointed at the data tile (as the
//   unary SFPU path leaves it at its tile) and RWC.Dst non-zero; the
//   enclosing `_llk_math_eltwise_sfpu_done_` clears RWC.Dst and the next
//   `_llk_math_eltwise_sfpu_start_` rewrites the dest base.
//
// BCAST_COL normalize path:
//   LREG0/LREG2 and LREG7/LREG1 carry the two interleaved broadcast chains
//   (mean and inverse standard deviation); LREG1 is LREG_DATA0 and is only
//   loaded once the chains have finished. It replays the block that
//   sfpu_bcast_col_init() records into slots [0, 16), so like add/sub/mul it
//   needs that init re-run after any op that rewrites the SFPU replay slots;
//   see the normalize section below.
//
// BCAST_SCALAR normalize path:
//   LREG6/LREG7 retain the negated scalar mean and inverse standard deviation for the
//   tile traversal. This invalidates the BCAST_COL lane mask in LREG6, so a
//   subsequent BCAST_COL operation must call sfpu_bcast_col_init() again.
//   LREG14 (programmable constant 2) is clobbered while extracting each scalar.
//   A subsequent SFPU operation that uses this constant must run its init first.
//   The band body is recorded into replay slots [0, 16) per call. That
//   overwrites the BCAST_COL broadcast program (the sfpu_bcast_col_init()
//   re-run that the LREG6 clobber already requires restores it) and any other
//   SFPU replay program recorded at init (reduce, cumsum, fast exp, ...),
//   whose op must re-run its init after a scalar normalize call. ADDR_MOD_6 is
//   not touched. Like the ROW path, the call leaves the dest base register at
//   the data tile and RWC.Dst non-zero.

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

// Replay-buffer slots. SFPU code owns [0, math::replay_buf_offset); the upper
// half belongs to FPU programs recorded at init (matmul, reduce,
// transpose_dest) and must never be written here. The ROW band body fills the
// whole SFPU half, so it overwrites the COL broadcast program: a COL op after
// a ROW op needs sfpu_bcast_col_init() again (it needs it anyway, since ROW
// also clobbers the LREG6 lane mask).
constexpr std::uint32_t REPLAY_SLOT_BROADCAST = 0;  // col-0 -> all-cols broadcast
constexpr std::uint32_t REPLAY_LEN_BROADCAST = 11;  // see _record_norm_dual_broadcast_replay_
// Whether the ROW band body advances RWC.Dst from its last SFPSTORE through
// ADDR_MOD_6 (programmed per call) instead of a trailing INCRWC. A throttled
// matmul (MM_THROTTLE > 0) programs ADDR_MOD_6 for its own MOP only at init,
// so in such a kernel the ROW path must not touch it.
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
constexpr bool ROW_BAND_STORE_INCR = false;
#else
constexpr bool ROW_BAND_STORE_INCR = true;
#endif
constexpr std::uint32_t REPLAY_SLOT_ROW_BAND = 0;                             // BCAST_ROW band body
constexpr std::uint32_t REPLAY_LEN_ROW_BAND = ROW_BAND_STORE_INCR ? 15 : 16;  // see _record_row_bcast_band_replay_
// BCAST_SCALAR normalize band body, recorded per call. It overwrites the COL
// init's broadcast program, which the scalar normalize already invalidates by
// clobbering LREG_MASK (sfpu_bcast_col_init() must be re-run after it).
constexpr std::uint32_t REPLAY_SLOT_SCALAR_NORM_BAND = 0;
constexpr std::uint32_t REPLAY_LEN_SCALAR_NORM_BAND = 16;  // see _record_scalar_normalize_band_replay_
static_assert(
    REPLAY_SLOT_BROADCAST + REPLAY_LEN_BROADCAST <= math::replay_buf_offset &&
        REPLAY_SLOT_ROW_BAND + REPLAY_LEN_ROW_BAND <= math::replay_buf_offset &&
        REPLAY_SLOT_SCALAR_NORM_BAND + REPLAY_LEN_SCALAR_NORM_BAND <= math::replay_buf_offset,
    "SFPU bcast replay programs must stay in the SFPU half of the replay buffer");

// SFPLOAD / SFPSTORE carry a 10-bit dest address that wraps; relative
// (possibly "negative") tile offsets are formed modulo this.
constexpr std::uint32_t DEST_ADDR_MASK = 0x3FF;

constexpr std::uint32_t dest_addr_wrap(std::uint32_t addr) { return addr & DEST_ADDR_MASK; }

// INCRWC's dest increment field is 4 bits; a 16-row face jump is two steps.
constexpr std::uint32_t INCRWC_FACE_HALF_STEP = 8;

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

// The BCAST_COL add/sub/mul broadcast program is the first
// REPLAY_LEN_BROADCAST slots of the normalize block that the COL init records
// (_record_norm_dual_broadcast_replay_): the mask (SFPAND LREG_BCAST &=
// LREG_MASK, zeroing SFPU cols 1..7 bitwise) + stage 1 (ROR by 1, add) +
// stage 2 (ROR by 2, add) on LREG_BCAST / LREG_TMP. Stage 3 (ROR by 4) is
// emitted inline by _broadcast_stage3_with_data_prefetch_() because it
// interleaves per-slot SFPLOADs into its SFPSHFT2 latency slots (data
// addresses are per-call so can't be recorded).

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
// arithmetic (mask AND, three rotate-adds, negate, add, multiply) and
// its order are unchanged, so results are bit-identical to the serial form.
//
// Dest addressing is RWC-relative so that the per-band body is a sequence of
// immediates: DEST_TARGET_REG_CFG_MATH_Offset is pointed at the data tile for
// the duration of the call and RWC.Dst walks the 8 row bands (INCRWC 4 per
// band, +16 at the face-pair boundary). The statistics tiles are addressed
// through 10-bit wrapped offsets relative to the data tile (the SFPU dest
// address adder is 10 bits wide, so a negative offset wraps correctly).
//
// The mean mask + stage 1 + stage 2 + first half of stage 3 of the interleaved
// chains (16 ops, no addresses) are recorded into replay slots [0, 16) by the
// COL init. This stays inside the SFPU half of the replay buffer (FPU kernels
// own [replay_buf_offset, 32)). The first REPLAY_LEN_BROADCAST slots double as
// the add/sub/mul broadcast program: the mean chain on LREG_BCAST / LREG_TMP,
// with the inv_std chain in what would otherwise be SFPNOP slots (it writes
// only LREG7 and LREG1 = LREG_DATA0, both dead at that point of the
// add/sub/mul path). That is why the inv_std mask is applied before the
// replay: an SFPAND may not follow an SFPSHFT2 ROR1 directly, so it cannot
// take the slot after the first shift.

constexpr std::uint32_t LREG_NORM_MEAN = LREG_BCAST;
constexpr std::uint32_t LREG_NORM_MEAN_TMP = LREG_TMP;
constexpr std::uint32_t LREG_NORM_INV_STD = p_sfpu::LREG7;
constexpr std::uint32_t LREG_NORM_INV_STD_TMP = LREG_DATA0;  // loaded only after the broadcast
static_assert(
    LREG_NORM_INV_STD != LREG_MASK && LREG_NORM_INV_STD != LREG_BCAST && LREG_NORM_INV_STD != LREG_TMP &&
        LREG_NORM_INV_STD != LREG_DATA1 && LREG_NORM_INV_STD != LREG_DATA2 && LREG_NORM_INV_STD != LREG_DATA3,
    "the inv_std chain may only write registers the COL add/sub/mul path does not read before writing");

constexpr std::uint32_t REPLAY_SLOT_NORM_BROADCAST = REPLAY_SLOT_BROADCAST;
constexpr std::uint32_t REPLAY_LEN_NORM_BROADCAST = 16;
static_assert(
    REPLAY_SLOT_NORM_BROADCAST + REPLAY_LEN_NORM_BROADCAST <= ckernel::math::replay_buf_offset,
    "SFPU replay window must stay below the FPU partition of the replay buffer");
static_assert(
    REPLAY_LEN_NORM_BROADCAST >= REPLAY_LEN_BROADCAST, "the normalize block must cover the COL broadcast program");

// The band body addresses the band's 4 col groups with the COL_GROUP_OFFSETS
// immediates relative to RWC.Dst (the left face's band row); the same
// immediates serve the lower half because RWC.Dst is moved to the face 2/3
// pair there.
inline std::uint32_t _norm_rel_tile_offset_(std::uint32_t dst_index, std::uint32_t dst_index_data) {
    return dest_addr_wrap((dst_index - dst_index_data) * DEST_TILE_SIZE_RAW);
}

// Record the interleaved broadcast block (16 ops). Precondition at replay:
// LREG_NORM_MEAN holds the freshly loaded col-0 slot, LREG_NORM_INV_STD the
// masked one. Replaying slots 1..11 alone is the add/sub/mul broadcast
// program; every SFPADD / SFPSHFT2 result is first read two slots later, and
// no SFPAND follows a shift.
//
//    1  SFPAND    MEAN    &= MASK
//    2  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 1
//    3  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//    4  SFPADD    MEAN    += MEAN_TMP
//    5  SFPADD    INV_STD += INV_STD_TMP
//    6  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 2
//    7  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//    8  SFPSHFT2  MEAN_TMP    = ROR1(MEAN_TMP)
//    9  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD_TMP)
//   10  SFPADD    MEAN    += MEAN_TMP
//   11  SFPADD    INV_STD += INV_STD_TMP
//   12  SFPSHFT2  MEAN_TMP    = ROR1(MEAN)           stage 3, first half
//   13  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD)
//   14  SFPSHFT2  MEAN_TMP    = ROR1(MEAN_TMP)
//   15  SFPSHFT2  INV_STD_TMP = ROR1(INV_STD_TMP)
//   16  SFPSHFT2  MEAN_TMP    = ROR1(MEAN_TMP)
inline void _record_norm_dual_broadcast_replay_() {
    lltt::record(REPLAY_SLOT_NORM_BROADCAST, REPLAY_LEN_NORM_BROADCAST);

    TTI_SFPAND(0, LREG_MASK, LREG_NORM_MEAN, 0);

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
    TTI_SFPSHFT2(0, LREG_NORM_MEAN_TMP, LREG_NORM_MEAN_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

// Load both col-0 slots of the current band, mask inv_std, run the recorded
// block and the rest of stage 3, and leave -mean in LREG_NORM_MEAN and
// inv_std in LREG_NORM_INV_STD, both ready to be read by the next instruction.
sfpi_inline void _norm_dual_broadcast_band_(std::uint32_t mean_rel, std::uint32_t inv_std_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(LREG_NORM_MEAN, IM, ADDR_MOD_3, mean_rel);
    TT_SFPLOAD(LREG_NORM_INV_STD, IM, ADDR_MOD_3, inv_std_rel);
    TTI_SFPAND(0, LREG_MASK, LREG_NORM_INV_STD, 0);
    lltt::replay(REPLAY_SLOT_NORM_BROADCAST, REPLAY_LEN_NORM_BROADCAST);
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
    TTI_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[0]);
    TTI_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[1]);
    TTI_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[2]);
    TTI_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[3]);
}

// Same for a second in-place tile at a wrapped offset from the data tile.
sfpi_inline void _norm_centre_scale_store_band_rel_(std::uint32_t tile_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t slot0 = dest_addr_wrap(tile_rel + COL_GROUP_OFFSETS[0]);
    const std::uint32_t slot1 = dest_addr_wrap(tile_rel + COL_GROUP_OFFSETS[1]);
    const std::uint32_t slot2 = dest_addr_wrap(tile_rel + COL_GROUP_OFFSETS[2]);
    const std::uint32_t slot3 = dest_addr_wrap(tile_rel + COL_GROUP_OFFSETS[3]);
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
    TTI_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[0]);
    TTI_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[1]);
    TTI_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[2]);
    TTI_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[3]);
}

// Point the math dest offset at the data tile.
// RWC.Dst is 0 on entry (the SFPU wrappers clear it) and ends at 48.
inline void _norm_col_tile_begin_(std::uint32_t dst_index_data) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index_data);
}

// Restore the dest offset programmed by _llk_math_eltwise_sfpu_start_ once
// the in-flight stores have drained (config writes must not overtake them).
inline void _norm_col_tile_end_() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
}

// Advance RWC.Dst from the end of the face 0/1 band run (16) to face 2/3 (32).
inline void _norm_col_next_face_pair_() {
    static_assert(FACE2_BASE - FACE1_BASE == 2 * INCRWC_FACE_HALF_STEP, "two INCRWCs span one face");
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);
}

inline void _calculate_sfpu_normalize_bcast_col_full_tile_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_out,
    std::uint32_t dst_index_inv_std) {
    // The result is written in place (the compute API always passes the data tile as out).
    LLK_ASSERT(
        dst_index_out == dst_index_data,
        "_calculate_sfpu_normalize_bcast_col_full_tile_: dst_index_out must equal dst_index_data");
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
    std::uint32_t dst_index_out,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_inv_std) {
    // The result is written in place (the compute API always passes the data tile as out).
    LLK_ASSERT(
        dst_index_out == dst_index_data,
        "_calculate_sfpu_residual_normalize_bcast_col_full_tile_: dst_index_out must equal dst_index_data");
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t residual_rel = _norm_rel_tile_offset_(dst_index_residual, dst_index_data);
    const std::uint32_t mean_rel = _norm_rel_tile_offset_(dst_index_mean, dst_index_data);
    const std::uint32_t inv_std_rel = _norm_rel_tile_offset_(dst_index_inv_std, dst_index_data);
    const std::uint32_t res_slot0 = dest_addr_wrap(residual_rel + COL_GROUP_OFFSETS[0]);
    const std::uint32_t res_slot1 = dest_addr_wrap(residual_rel + COL_GROUP_OFFSETS[1]);
    const std::uint32_t res_slot2 = dest_addr_wrap(residual_rel + COL_GROUP_OFFSETS[2]);
    const std::uint32_t res_slot3 = dest_addr_wrap(residual_rel + COL_GROUP_OFFSETS[3]);

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

// Broadcast element zero of the mean and inv_std tiles to all 32 lanes:
// LREG_SCALAR_NEG_MEAN := -mean[0][0], LREG_SCALAR_INV_STD := inv_std[0][0].
// The two rotate-and-add chains are interleaved, so each instruction's
// consumer is two slots later and no SFPNOP is needed (SFPADD and SFPSHFT2
// SUBVEC_SHFLROR1 only forbid reading their result on the next cycle, and
// neither is on SFPSHFT2's next-cycle exclusion list).
inline void _broadcast_scalars_from_dest_(std::uint32_t mean_addr, std::uint32_t inv_std_addr) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    constexpr std::uint32_t M = p_sfpu::LREG4, M_TMP = p_sfpu::LREG2;
    constexpr std::uint32_t I = LREG_SCALAR_INV_STD, I_TMP = p_sfpu::LREG3;
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, mean_addr);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, inv_std_addr);
    // The unused statistic-tile lanes can contain infinities after rsqrt.
    // Predicated assignment is required here: multiplying those lanes by a
    // zero mask would produce NaNs rather than zeros.
    // As in _build_lane_mask_col0_, shifting 2*lane_id left by 28 leaves zero
    // only in SFPU column 0.
    TTI_SFPMOV(0, p_sfpu::LTILEID, LREG_TMP, 0);
    TTI_SFPSHFT(28, 0, LREG_TMP, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPSETCC(0, LREG_TMP, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    // SFPCONFIG vertically broadcasts the first eight lanes of LREG0 into
    // LREG14, which puts scalar lane zero in every 8-lane sub-vector; the
    // rotate-and-add chains below fill each one.
    // ttsim v1.10.5 permits non-uniform source lanes only for LREG14.
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG14, M, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG14, I, 0);
    // Rotate by 1, 2, 4 and add: lane 0 reaches all 8 lanes of each sub-vector.
    TTI_SFPSHFT2(0, M, M_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, I, I_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(M, p_sfpu::LCONST_1, M_TMP, M, 0);
    TTI_SFPADD(I, p_sfpu::LCONST_1, I_TMP, I, 0);
    TTI_SFPSHFT2(0, M, M_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, I, I_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, M_TMP, M_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, I_TMP, I_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(M, p_sfpu::LCONST_1, M_TMP, M, 0);
    TTI_SFPADD(I, p_sfpu::LCONST_1, I_TMP, I, 0);
    TTI_SFPSHFT2(0, M, M_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, I, I_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    for (int k = 0; k < 3; k++) {
        TTI_SFPSHFT2(0, M_TMP, M_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, I_TMP, I_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    }
    TTI_SFPADD(M, p_sfpu::LCONST_1, M_TMP, M, 0);
    TTI_SFPADD(I, p_sfpu::LCONST_1, I_TMP, I, 0);
    // Negate after the sums (not before) so that -0.0 / +0.0 match the
    // reference (x - mean) exactly.
    TTI_SFPMOV(0, M, LREG_SCALAR_NEG_MEAN, 1 /* SFPMOV_MOD1_NEGATE */);
}

// One row band of the scalar normalize: 4 loads, 4 adds, 4 muls, 4 stores,
// all addressed relative to RWC.Dst with the dest base register pointed at
// the data tile, so the body is identical for the 8 bands of a tile. It is
// recorded into the replay buffer while band 0 executes (Exec, no SFPU cost)
// and replayed for bands 1..7; the RISC then issues a REPLAY and an INCRWC per
// band instead of 8 runtime TT_SFPLOAD/TT_SFPSTORE encodes (3 RISC
// instructions each), which made this path RISC-bound (330 RISC vs 186 SFPU
// instructions per tile). `out_rel` (the data -> out tile distance, 0 for
// the in-place API) is baked into the store immediates when recording.
// Exactly REPLAY_LEN_SCALAR_NORM_BAND instructions.
inline void _record_scalar_normalize_band_replay_(std::uint32_t out_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    lltt::record<lltt::Exec>(REPLAY_SLOT_SCALAR_NORM_BAND, REPLAY_LEN_SCALAR_NORM_BAND);

    TTI_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[0]);
    TTI_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[1]);
    TTI_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[2]);
    TTI_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[3]);

    TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA0, 0);
    TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA1, 0);
    TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA2, 0);
    TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_SCALAR_NEG_MEAN, LREG_DATA3, 0);

    TTI_SFPMUL(LREG_DATA0, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA0, 0);
    TTI_SFPMUL(LREG_DATA1, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA1, 0);
    TTI_SFPMUL(LREG_DATA2, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA2, 0);
    TTI_SFPMUL(LREG_DATA3, LREG_SCALAR_INV_STD, p_sfpu::LCONST_0, LREG_DATA3, 0);

    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[0]));
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[1]));
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[2]));
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[3]));
}

inline void _calculate_sfpu_normalize_bcast_scalar_full_tile_(
    std::uint32_t dst_index_data,
    std::uint32_t dst_index_mean,
    std::uint32_t dst_index_out,
    std::uint32_t dst_index_inv_std) {
    // Dest base := data tile, RWC.Dst := 0. The statistics and output tiles
    // are addressed relative to it; the SFPU dest address is 10 bits and
    // wraps, so a tile below the data tile is reached with a wrapped offset.
    // As on the ROW path, the call leaves the dest base at the data tile and
    // RWC.Dst non-zero for the caller's `_done_` / next `_start_` to reset.
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index_data);
    math::clear_dst_reg_addr();

    const std::uint32_t mean_rel = dest_addr_wrap((dst_index_mean - dst_index_data) * DEST_TILE_SIZE_RAW);
    const std::uint32_t inv_std_rel = dest_addr_wrap((dst_index_inv_std - dst_index_data) * DEST_TILE_SIZE_RAW);
    const std::uint32_t out_rel = dest_addr_wrap((dst_index_out - dst_index_data) * DEST_TILE_SIZE_RAW);

    // The statistics are scalar and invariant across all eight row bands.
    // Broadcast each once and retain -mean/inv_std in LREG6/LREG7 for the
    // complete tile traversal. LREG6 deliberately borrows the BCAST_COL mask.
    _broadcast_scalars_from_dest_(mean_rel, inv_std_rel);

    // Upper half (faces 0/1): band 0 records + executes, bands 1..3 replay.
    _record_scalar_normalize_band_replay_(out_rel);
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ ROW_BAND_STRIDE, /*rwc_b*/ 0, /*rwc_a*/ 0);
    for (std::uint32_t band = 1; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        lltt::replay(REPLAY_SLOT_SCALAR_NORM_BAND, REPLAY_LEN_SCALAR_NORM_BAND);
        TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ ROW_BAND_STRIDE, /*rwc_b*/ 0, /*rwc_a*/ 0);
    }

    // RWC.Dst: FACE1_BASE -> FACE2_BASE (INCRWC's increment is 4 bits wide).
    static_assert(FACE2_BASE - FACE1_BASE == 2 * INCRWC_FACE_HALF_STEP, "two INCRWCs span one face");
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);

    // Lower half (faces 2/3): bands 0..3 replay.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        lltt::replay(REPLAY_SLOT_SCALAR_NORM_BAND, REPLAY_LEN_SCALAR_NORM_BAND);
        TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ ROW_BAND_STRIDE, /*rwc_b*/ 0, /*rwc_a*/ 0);
    }
}

// ============================================================================
// BCAST_ROW: driver (SFPTRANSP + in-place, band body in the replay buffer)
// ============================================================================
//
// Every dest address in this path is an immediate relative to RWC.Dst, with
// the dest base register pointed at the data tile. That makes the per-band
// body band-invariant (it ends by advancing RWC.Dst to the next band), so it
// is recorded into the replay buffer once per full-tile call - executing
// band 0 as it records - and replayed for the remaining 7 bands. The band
// loop then costs the RISC one REPLAY per band instead of 8 encoded `TT_`
// instructions, which is what made this path RISC-bound (275 RISC vs 124
// SFPU instructions per tile).
//
// The recording is per call rather than per init because the init is shared
// across ADD / SUB / MUL (`sfpu_bcast_row_init`) and the body depends on the
// op. Recording with Exec costs no SFPU time: band 0 runs while it records.
//
// Setup (once per full-tile call, hoisted out of the per-band loop):
//   0. Dest base := data tile (SETC16); RWC.Dst := 0. The previous SFPU op's
//      `_llk_math_eltwise_sfpu_done_` waited for the SFPU, and the SETC16 is
//      ordered before the SFPLOAD/SFPSTOREs that follow it. Load bcast tile's
//      4 col-groups (face 0, row band 0)
//      into LREG4..7, addressed relative to the data tile. The dest address
//      is 10 bits and wraps, so a bcast tile below the data tile is reached
//      with a "negative" (wrapped) offset.
//
// Per band (8 bands per tile; band 0 recorded + executed, 1..7 replayed):
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
//   5. Store LREG0..3 to the 4 col-groups of the output band. The output
//      tile's distance from the data tile is the only per-call input of the
//      body; it is baked into the 4 store immediates when recording.
//   6. RWC.Dst += ROW_BAND_STRIDE (the last store's ADDR_MOD_6, or an INCRWC
//      under MM_THROTTLE), so the next band uses the same
//      immediates. After 4 bands RWC.Dst sits at FACE1_BASE; the driver
//      bumps it to FACE2_BASE (face 1 was the right half of the upper bands)
//      before the lower 4 bands.

template <BinaryOp BINOP>
inline void _record_row_bcast_band_replay_(std::uint32_t out_rel) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // Exec: band 0 executes as it records; bands 1..7 replay the copy.
    lltt::record<lltt::Exec>(REPLAY_SLOT_ROW_BAND, REPLAY_LEN_ROW_BAND);

    // (1) Load data's 4 col-groups into LREG0..3, relative to RWC.Dst. Bcast
    //     (LREG4..7) was pre-loaded once by the caller and survives across bands.
    TTI_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[0]);
    TTI_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[1]);
    TTI_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[2]);
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, COL_GROUP_OFFSETS[3]);

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

    // (5) Store LREG0..3 to the 4 col-groups of the output band. `out_rel` is
    //     the per-call data -> out tile distance (0 for the in-place API).
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[0]));
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[1]));
    TT_SFPSTORE(p_sfpu::LREG2, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[2]));
    // (6) Next band: the last store post-increments RWC.Dst through ADDR_MOD_6
    //     (named ADDR_MOD_2 under the SFPU addr-mod base), or an INCRWC does.
    if constexpr (ROW_BAND_STORE_INCR) {
        TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_2, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[3]));
    } else {
        TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_3, dest_addr_wrap(out_rel + COL_GROUP_OFFSETS[3]));
        TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ ROW_BAND_STRIDE, /*rwc_b*/ 0, /*rwc_a*/ 0);
    }
}

template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_row_full_tile_(
    std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out) {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // Dest base := data tile, RWC.Dst := 0. From here on every address is a
    // band-relative immediate (see the driver comment above).
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index_data);
    math::clear_dst_reg_addr();
    if constexpr (ROW_BAND_STORE_INCR) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ROW_BAND_STRIDE}}.set(ADDR_MOD_6);
    }

    const std::uint32_t bcast_rel = dest_addr_wrap((dst_index_bcast - dst_index_data) * DEST_TILE_SIZE_RAW);
    const std::uint32_t out_rel = dest_addr_wrap((dst_index_out - dst_index_data) * DEST_TILE_SIZE_RAW);

    // Hoisted bcast load: bcast row 0 lives in face 0 row band 0 and is
    // identical for every output band, so we load its 4 col-groups exactly
    // once. The per-band binop is structured to leave LREG4..7 unchanged,
    // so the second SFPTRANSP per band restores them to this face-major
    // layout.
    TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_3, dest_addr_wrap(bcast_rel + FACE0_BASE + COL_GROUP_OFFSETS[0]));
    TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_3, dest_addr_wrap(bcast_rel + FACE0_BASE + COL_GROUP_OFFSETS[1]));
    TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_3, dest_addr_wrap(bcast_rel + FACE0_BASE + COL_GROUP_OFFSETS[2]));
    TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_3, dest_addr_wrap(bcast_rel + FACE0_BASE + COL_GROUP_OFFSETS[3]));

    // Upper half (face 0/1): band 0 records + executes, bands 1..3 replay.
    _record_row_bcast_band_replay_<BINOP>(out_rel);
    for (std::uint32_t band = 1; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        lltt::replay(REPLAY_SLOT_ROW_BAND, REPLAY_LEN_ROW_BAND);
    }

    // RWC.Dst: FACE1_BASE -> FACE2_BASE.
    static_assert(FACE2_BASE - FACE1_BASE == 2 * INCRWC_FACE_HALF_STEP, "two INCRWCs span one face");
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);
    TTI_INCRWC(/*rwc_cr*/ 0, /*rwc_d*/ INCRWC_FACE_HALF_STEP, /*rwc_b*/ 0, /*rwc_a*/ 0);

    // Lower half (face 2/3): bands 0..3 replay.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++) {
        lltt::replay(REPLAY_SLOT_ROW_BAND, REPLAY_LEN_ROW_BAND);
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

        // Record the normalize broadcast block into replay slot 0; its first
        // REPLAY_LEN_BROADCAST slots are the add/sub/mul broadcast program.
        _record_norm_dual_broadcast_replay_();
    }
    // BCAST_ROW: nothing persists from the init. Each full-tile call records
    // its band body into SFPU replay slots 0..15, clobbers LREG4..7 and
    // (without MM_THROTTLE) reprograms ADDR_MOD_6, so any SFPU op that sets
    // that state at init (COL bcast, reduce, fast exp, abs, i0, ...) must
    // re-run its init after a ROW call.
}

}  // namespace sfpu
}  // namespace ckernel
