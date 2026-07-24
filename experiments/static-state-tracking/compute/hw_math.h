// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// MATH-engine hardware configurations.
//
//   math_hw_cfg              — one-shot ALU/format programming (startup).
//   math_pack_sync_cfg       — claim the whole DST at startup.
//   math_remap_cfg           — enable the stride-16 DST read the untilizer wants.
//   math_a2d_cfg             — program the MOV_8_ROWS datacopy MOP.
//   math_a2d                 — copy one tile A->DST[i] (per tile).
//   math_wait_for_dest_available / math_dest_section_done — tile_regs_* (MATH).

#ifndef SST_COMPUTE_HW_MATH_H
#define SST_COMPUTE_HW_MATH_H

#include <cstdint>

#include "defs.h"

#ifdef TRISC_MATH

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "experiments/static-state-tracking/inc/state.h"  // sst::TileConfig

namespace sst {
namespace compute {
namespace hw {

using namespace ckernel;

// One-shot math configure.
template <typename Traits>
inline void math_hw_cfg() {
    constexpr bool fp32 = Traits::fp32_dest_acc;

    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(fp32);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(fp32);
}

// Claim the whole DST for MATH at startup (blocks until previous packs drain).
template <typename Traits>
inline void math_pack_sync_cfg() {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    if constexpr (Traits::dst_sync == sst::DstSyncMode::SyncFull) {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
    } else {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    }
    reset_dest_offset_id();
    math::set_dest_section_base<DstStart::StartZero>();
}

// Enable/disable the DEST stride-16 remap the untilize packer consumes. Must be
// enabled BEFORE the producer writes DST (Blackhole ordering constraint).
inline void math_remap_cfg(bool remap_enable) {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap_enable);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(remap_enable);
}

// Program the A2D datacopy MOP: addrmods (idle / per-row / MOV_8_ROWS) + a
// MOV_8_ROWS MOP (num_faces outer, 16>>3=2 inner) + dvalid/counter epilogue.
template <typename Traits>
inline void math_a2d_cfg(const sst::TileConfig& tile_config) {
    constexpr bool fp32 = Traits::fp32_dest_acc;

    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 0}, .dest = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_2);

    const std::uint32_t outerloop = tile_config.num_faces;
    const std::uint32_t innerloop = 16u >> 3;
    if constexpr (fp32) {
        ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    } else {
        ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Copy one tile A->DST at `dst_tile_index`.
inline void math_a2d(std::uint32_t dst_tile_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_tile_index);
    ckernel::ckernel_template::run();
    math::clear_dst_reg_addr();
}

// tile_regs_acquire (MATH side): wait until DST is free to write.
inline void math_wait_for_dest_available() { math::math_dest_wait(); }

// tile_regs_commit (MATH side): hand the written DST section to PACK.
template <typename Traits>
inline void math_dest_section_done() {
    math::set_math_semaphores();
    if constexpr (Traits::dst_sync == sst::DstSyncMode::SyncHalf) {
        math_sync_tile_dst_index = 0;
        math::dest_section_flip();
    }
}

}  // namespace hw
}  // namespace compute
}  // namespace sst

#endif  // TRISC_MATH

#endif  // SST_COMPUTE_HW_MATH_H
