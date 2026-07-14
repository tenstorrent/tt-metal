// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// UNPACK-engine hardware configurations.
//   unpack_hw_cfg              — one-shot format/tile-descriptor programming (startup).
//   unpack_datacopy_face_cfg — within-face transpose + per-face x_end (keyed on tile_config).
//   unpack_datacopy_mop_cfg    — the SrcA UNPACR datacopy MOP (keyed on op mode).
//   unpack_a                   — issue one tile's unpack (per copied tile, steady state).

#ifndef SST_COMPUTE_HW_UNPACK_H
#define SST_COMPUTE_HW_UNPACK_H

#include <cstdint>

#include "defs.h"

#ifdef TRISC_UNPACK

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_defs.h"

#include "experiments/static-state-tracking/inc/state.h"  // sst::TileConfig

namespace sst {
namespace compute {
namespace hw {

using namespace ckernel;
using namespace ckernel::unpacker;

// One-shot unpack configure.
inline void unpack_hw_cfg(const sst::TileConfig& tile_config) {
    constexpr bool fp32 = (DST_ACCUM_MODE != 0);
    const std::uint32_t df = tile_config.data_format;
    const std::uint32_t tile_size_bytes = sst::tensor::tile_size_bytes_from_tile_config(tile_config);

    configure_unpack_AB<fp32>(
        df,
        df,
        df,
        df,
        tile_config.face_r_dim,
        tile_config.face_r_dim,
        /*transpose_xy_srca_en=*/false,
        tile_config.num_faces,
        tile_config.num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_bytes), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_bytes), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
}

// --- SrcA-copy configure, split into two field-keyed sub-steps so the tracker
// can reprogram only what changed:
//   unpack_datacopy_face_cfg — within-face transpose + per-face x_end. Depends on
//                            the tile geometry (face_r_dim), i.e. the tracked tile_config.
//   unpack_datacopy_mop_cfg    — the UNPACR SrcA MOP (outer loop = num_faces). Depends
//                            on the op mode (datacopy) and num_faces.
inline void unpack_datacopy_face_cfg(const sst::TileConfig& tile_config) {
    // within_face_16x16_transpose = 0
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    // Unpack the entire face on UNP_A.
    config_unpacker_x_end<p_setadc::UNP_A>(tile_config.face_r_dim);
}

inline void unpack_datacopy_mop_cfg(const sst::TileConfig& tile_config) {
    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(
        SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid =
        TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    ckernel_template tmp(tile_config.num_faces, 1 /*inner_loop*/, unpack_srcb_set_dvalid);
    tmp.set_start_op(unpack_srca);
    tmp.program();
}

// Issue one tile's unpack: set the SrcA L1 base address for the current cfg context, run the per-face UNPACR MOP.
inline void unpack_a(std::uint32_t l1_addr_16B) {
    const std::uint32_t address = l1_addr_16B - 1;

    // Reset both unpackers' Z/W address counters to 0 so this tile's unpack starts at the tile base.
    TTI_SETADCZW(0b011 /*UNP_A|UNP_B*/, 0 /*Ch1_W*/, 0 /*Ch1_Z*/, 0 /*Ch0_W*/, 0 /*Ch0_Z*/, 0b1111 /*write all Z/W*/);

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    const std::uint32_t upk0_reg =
        (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg] = address;

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel::ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

// --- Matmul UNPACK recipe (prototype), transcribed from the LLK matmul unpack path
// (llk_unpack_AB_matmul.h) for the FIXED case: full 32x32 tiles, ct=rt=kt=1 (single
// tile, reuse_a=true), no partial face, no broadcast. Convention: in0 -> SrcB (SEC0),
// in1 -> SrcA (SEC1). The AB MOP streams SrcA (in1) with a per-tile base advance; the
// execute unpacks SrcB (in0) directly, then runs the MOP.
//   unpack_matmul_mop_cfg — the two-context SrcA replay + unpack template.
//   unpack_matmul         — set SrcA/SrcB L1 bases, unpack SrcB, run the SrcA MOP.
inline void unpack_matmul_mop_cfg(const sst::TileConfig& /*tile_config*/) {
    // Whole-tile unpack setup (no partial face): re-enable face transpose, reset the
    // Z/W counters, and widen the X-dim so ONE UNPACR unpacks the full 32x32 tile.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    const std::uint32_t x_end = 4 /*num_faces*/ * FACE_R_DIM * FACE_C_DIM - 1;  // whole 32x32 tile
    TT_SETADCXX(p_setadc::UNP_A, x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, x_end, 0x0);
    TT_SETDMAREG(0, LOWER_HALFWORD(1), 0, LO_16(p_gpr_unpack::KT_DIM));  // kt_dim = 1

    // Per-context (6 instrs each): unpack a whole SrcA tile (in1, set dvalid), then
    // advance the SrcA L1 base by one tile (matmul-style RDCFG/ADDDMAREG/WRCFG).
    constexpr std::uint32_t prog_len = 12;
    constexpr std::uint32_t run_len = 6;
    load_replay_buf(0, prog_len, [] {
        // context 0
        TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
        // context 1
        TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
    });
    // A0 = context-0 replay, skipA = context-1 replay; the TT_MOP zmask selects by context.
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false /*srcB*/,
        false /*halo*/,
        lltt::replay_insn(0, run_len),
        0,
        0,
        0,
        lltt::replay_insn(run_len, run_len),
        0,
        0);
    tmp.program();
}

inline void unpack_matmul(std::uint32_t in0_l1_addr_16B, std::uint32_t in1_l1_addr_16B) {
    const std::uint32_t addr_in0 = in0_l1_addr_16B - 1;  // in0 -> SrcB (SEC1)
    const std::uint32_t addr_in1 = in1_l1_addr_16B - 1;  // in1 -> SrcA (SEC0)

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    // in1 -> SrcA (SEC0), in0 -> SrcB (SEC1), for the current config context.
    const std::uint32_t sec0_reg =
        (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    const std::uint32_t sec1_reg =
        (unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;
    cfg[sec0_reg] = addr_in1;  // in1 -> SrcA
    cfg[sec1_reg] = addr_in0;  // in0 -> SrcB

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    // Unpack in0 -> SrcB directly; the MOP then streams in1 -> SrcA.
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TT_MOP(0, 0 /*ct_dim-1*/, unp_cfg_context == 0 ? 0 : 0xff);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

}  // namespace hw
}  // namespace compute
}  // namespace sst

#endif  // TRISC_UNPACK

#endif  // SST_COMPUTE_HW_UNPACK_H
