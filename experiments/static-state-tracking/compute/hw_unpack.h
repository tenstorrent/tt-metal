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
template <typename Traits>
inline void unpack_hw_cfg(const sst::TileConfig& src_a, const sst::TileConfig& src_b) {
    constexpr bool fp32 = Traits::fp32_dest_acc;
    const std::uint32_t df_a = src_a.data_format;
    const std::uint32_t df_b = src_b.data_format;
    const std::uint32_t tile_size_bytes_a = sst::tensor::tile_size_bytes_from_tile_config(src_a);
    const std::uint32_t tile_size_bytes_b = sst::tensor::tile_size_bytes_from_tile_config(src_b);

    configure_unpack_AB<fp32>(
        df_a,              // unpA_src_format
        df_b,              // unpB_src_format
        df_a,              // unpA_dst_format
        df_b,              // unpB_dst_format
        src_a.face_r_dim,  // unpA_face_r_dim
        src_b.face_r_dim,  // unpB_face_r_dim
        /*transpose_xy_srca_en=*/false,
        src_a.num_faces,   // unpA_num_faces
        src_b.num_faces);  // unpB_num_faces

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_bytes_b), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_bytes_a), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
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

}  // namespace hw
}  // namespace compute
}  // namespace sst

#endif  // TRISC_UNPACK

#endif  // SST_COMPUTE_HW_UNPACK_H
