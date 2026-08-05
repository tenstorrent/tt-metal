// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight "weighted reduction" matmul for the DSA indexer SDPA op.
//
// Computes, per chunk:   out[1, 32] = weights[1, 8] x qk[8, 32]
//   out[p] = sum_{h=0..7} weights[h] * qk[h, p]      (p = 0..31)
//
// Realized with two TTI_MVMUL instructions (no matmul MOP / replay reprogram).
// On Blackhole MVMUL computes  Dst[i,j] += sum_k SrcB[i,k] * SrcA[k,j], so:
//   - SrcB holds the weights vector in row 0 (cols 0..7 = head weights, 8..15
//     zero). Rows 1..7 are zero, so only Dst row 0 is populated.
//   - SrcA holds the qk matrix; face0 = qk[0:8, 0:16], face1 = qk[0:8, 16:32]
//     sits one 16-row face above face0 in SrcA.
//   - MVMUL #1 (qk face0)  -> Dst rows 0..7,  cols 0..15  (face0, out[0:16])
//     then ADDR_MOD_0 advances SrcA +16 rows (-> face1) and Dst +16 rows.
//   - MVMUL #2 (qk face1)  -> Dst rows 16..23, cols 0..15 (face1, out[16:32]).
// So DEST logical row 0 == out[0:32] (face0 row0 cols0..15 + face1 row0
// cols16..31). Only logical row 0 is meaningful.
//
// DEST half: this op interleaves with the QK sdpa_custom_mm each chunk and runs
// on the second SyncHalf of the pair. Math and pack both target the
// framework-tracked dest half (get_dest_buffer_base() on the math side,
// SEC0_Offset on the pack side); these stay in sync via the tile_regs
// acquire/commit/release protocol. The packer ZEROACCs that half on
// dest-section-done, so the accumulating MVMULs always start from a cleared bank
// -- no manual clear or src-valid stall needed.
//
// The packer then writes DEST logical row 0 (two faces, one row each, single
// interface, two raw PACR -- no pack MOP) straight into the chunk's logical row of
// the `partial` tile (tiled layout) -- no row-major round-trip / tilize.

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_common.h"
#endif

namespace ckernel {

// Blackhole-only: the body is written against Blackhole SFPU/packer encodings. The LLK headers included
// above are arch-generic, so only the API surface needs gating.
#if defined(ARCH_BLACKHOLE)

// Re-establish the weighted-reduce unpack/math/pack config for one chunk.
//   weights_cb : [1, 32] tile (row 0 cols 0..7 = head weights)  -> SrcB
//   qk_cb      : [8, 32] tile (two 8x16 faces)                  -> SrcA
//   partial_cb : [32, 32] destination tile                      -> pack out
inline void weighted_reduce_init_short(
    const std::uint32_t weights_cb, const std::uint32_t qk_cb, const std::uint32_t partial_cb) {
    // SrcA <- qk (operandA), SrcB <- weights (operandB).
    reconfig_data_format<SrcOrder::Regular, true>(qk_cb, weights_cb);
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0)));
    UNPACK(TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0));
}

// Configure the weighted-reduce ADDR_MODs. These are constant across chunks, so
// call once outside the per-chunk loop (the QK matmul and regular pack init leave
// ADDR_MOD_6 (math) and ADDR_MOD_3 (pack) untouched).
//   MATH ADDR_MOD_6 : after MVMUL #1, advance SrcA by one 16-row face (-> qk
//                     face1) and DEST by 8 rows.
//   PACK ADDR_MOD_3 : between the two pack PACR, step the DEST read to face1.
// (MVMUL #2 uses ADDR_MOD_3 from sdpa_custom_mm_init; PACR #2 uses ADDR_MOD_1 from
// regular pack init.)
inline void weighted_reduce_addrmod_init() {
    MATH((addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
              .set(ADDR_MOD_6)));
    PACK((addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 0, .cr = 0},
        .y_dst = {.incr = 1, .clr = 0, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
        .z_dst = {.incr = 0, .clr = 0},
    }
              .set(ADDR_MOD_3)));
}

#ifdef TRISC_MATH
inline void weighted_reduce_math_impl() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Target tile 0 of the framework-tracked dest half (kept in sync with the
    // packer, which ZEROACCs this half on dest-section-done, so the accumulating
    // MVMULs start from a cleared bank).
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
    // MVMUL #1: qk face0 -> Dst row 0 (out[0:16]); then advance SrcA+16, Dst+16.
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_6, 0);
    // MVMUL #2: qk face1 -> Dst row 16 (out[16:32]); clear SrcA/SrcB.
    // ADDR_MOD_3 is set in sdpa_custom_mm_init
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
}
#endif

#ifdef TRISC_UNPACK
// Raw (no-MOP) replacement for llk_unpack_AB(qk, weights, 0, 0): load tile 0 of
// qk -> SrcA (both faces) and tile 0 of weights -> SrcB. One SrcB unpack is enough
// since the MVMULs only read SrcB cols 0..7 (weights face0). Three UNPACR replace
// the standard AB template MOP, with SetDatValid deferred to the last unpack of
// each source so both srcs go valid only once fully loaded.
inline void weighted_reduce_unpack_impl(const std::uint32_t weights_cb, const std::uint32_t qk_cb) {
    const std::uint32_t qk_id = get_operand_id(qk_cb);            // operandA -> SrcA
    const std::uint32_t weights_id = get_operand_id(weights_cb);  // operandB -> SrcB
    const std::uint32_t address_a = get_local_cb_interface(qk_id).fifo_rd_ptr - 1;
    const std::uint32_t address_b = get_local_cb_interface(weights_id).fifo_rd_ptr - 1;

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' counters
    // Wait for a free context, program SrcA/SrcB base addresses, release to unpacker.
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // qk face0 -> SrcA rows 0..15; AddrMode advances the L1 read + SrcA write to
    // face1, no dvalid yet.
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // weights -> SrcB row 0; sets SrcB dvalid.
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // qk face1 -> SrcA rows 16..31; sets SrcA dvalid.
    TTI_UNPACR(SrcA, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' counters
}
#endif

// Unpack qk + weights and run the two MVMULs. Call between tile_regs_acquire and
// tile_regs_commit.
inline void weighted_reduce(const std::uint32_t weights_cb, const std::uint32_t qk_cb) {
    UNPACK((weighted_reduce_unpack_impl(weights_cb, qk_cb)));
    MATH((weighted_reduce_math_impl()));
}

#ifdef TRISC_PACK
// Pack DEST logical row 0 (two faces, one row each) into the chunk's row
// of the `partial` tile. chunk maps to row (chunk % TILE_R_DIM) of
// tile (chunk / TILE_R_DIM). Packer L1 addresses are in 16B words
// (cb_addr_shift == 4): one face row = 16 bf16 = 32 B = 2 words; a 16x16 face = 32 words
inline void weighted_reduce_pack_impl(const std::uint32_t partial_cb, const std::uint32_t chunk) {
    constexpr std::uint32_t row_1x32_size_words = 4;  // 1x32 = 4 words
    const std::uint32_t tile = chunk / TILE_R_DIM;
    const std::uint32_t row = chunk % TILE_R_DIM;
    const std::uint8_t out_id = get_output_id(partial_cb);
    const std::uint32_t addr =
        get_output_tile_address<true, ckernel::PackMode::Default>(out_id, tile) + row * row_1x32_size_words;

    // ADDR_MOD_3 (pack) is configured once by weighted_reduce_addrmod_init().
    set_dst_write_addr(0);             // read DEST tile 0 (logical row 0 = result)
    program_packer_destination(addr);  // L1 write base for face0/face1 of this row
    // Raw (no-MOP) pack of DEST logical row 0 into the partial tile's row. PACR #1
    // (ADDR_MOD_3, set above) packs face0 and steps the DEST read to face1; PACR #2
    // (ADDR_MOD_1 from regular pack init) packs face1 and closes the tile (Last=1).
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_3,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        0);
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);  // reset z counters
}
#endif

// Pack the result. Call between tile_regs_wait and tile_regs_release.
inline void weighted_reduce_pack(const std::uint32_t partial_cb, const std::uint32_t chunk) {
    PACK((weighted_reduce_pack_impl(partial_cb, chunk)));
}

// Restore dataformats, and cfgs to what is needed for sdpa_custom_mm_block
inline void weighted_reduce_uninit(const std::uint32_t q_in_cb, const std::uint32_t k_in_cb) {
    // SrcA <- k_in (operandA), SrcB <- q_in (operandB).
    reconfig_data_format<SrcOrder::Regular, true>(k_in_cb, q_in_cb);
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1)));
    // UnpA unpacks full tiles
    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    UNPACK(TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0));
}

#endif

}  // namespace ckernel
