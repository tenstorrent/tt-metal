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

// One-shot math configure (srcA == srcB format). Float16_b: no int8 math, no
// fp32 dest.
inline void math_hw_cfg(const sst::TileConfig& tile_config) {
    constexpr bool fp32 = (DST_ACCUM_MODE != 0);
    const std::uint32_t df = tile_config.data_format;

    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);

    const std::uint32_t int8_math_enabled = (masked_data_format(df) == ckernel::to_underlying(DataFormat::Int8)) ||
                                            (df == ckernel::to_underlying(DataFormat::Int32));
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(fp32);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(fp32);
}

// Claim the whole DST for MATH at startup (blocks until previous packs drain).
inline void math_pack_sync_cfg() {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
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
inline void math_a2d_cfg(const sst::TileConfig& tile_config) {
    constexpr bool fp32 = (DST_ACCUM_MODE != 0);

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

// --- Matmul MATH recipe (prototype), transcribed from the LLK matmul math path
// (llk_math_matmul.h: matmul_configure_addrmod / matmul_configure_mop / _llk_math_matmul_)
// for the FIXED case: MATH_FIDELITY = HiFi2, full 32x32 tiles, ct_dim=rt_dim=1
// (single output tile), no transpose, num_faces=4. All conditionals resolved out.
//   math_matmul_mop_cfg — matmul addrmods + the 16-MVMUL full-tile MOP (2 fidelity phases).
//   math_matmul         — run the matmul MOP once, accumulating into DST[dst_tile_index].
inline void math_matmul_mop_cfg(const sst::TileConfig& /*tile_config*/) {
    // Matmul addrmods (HiFi2, full-tile, no-transpose): ADDR_MOD_0/1/2/4 step the
    // systolic SrcA/SrcB/Dest walk; ADDR_MOD_5 resets + advances the fidelity phase.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_0);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 1, .clr = 0}}
        .set(ADDR_MOD_5);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 0, .clr = 0, .cr = 1}, .dest = {.incr = 8}}.set(ADDR_MOD_1);
    addr_mod_t{.srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 32, .clr = 0, .cr = 1}, .dest = {.incr = 8}}
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1}}
        .set(ADDR_MOD_4);

    // The 16-MVMUL full-tile systolic sequence (recorded once, replayed per fidelity phase).
    constexpr std::uint32_t replay_len = 16;
    load_replay_buf(ckernel::math::replay_buf_offset, replay_len, [] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0A0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B0A0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);  // B0A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B2A0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B2A0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B2A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // B2A1
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B1A2
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B1A2
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B1A3
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);  // B1A3
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B3A2
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B3A2
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B3A3
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);  // reset srca/srcb/dest, advance fidelity phase
    });
    // HiFi2 -> inner_loops = 2 (two fidelity phases); CLR_A at the end (reuse_a).
    ckernel_template tmp(
        1 /*outer*/, 2 /*inner: fidelity phases*/, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_len));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
    tmp.program();

    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Multiply-accumulate one tile pair into DST[dst_tile_index] (SrcA=in1, SrcB=in0 already unpacked).
inline void math_matmul(std::uint32_t dst_tile_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_tile_index);
    ckernel::ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}

// tile_regs_acquire (MATH side): wait until DST is free to write.
inline void math_wait_for_dest_available() { math::math_dest_wait(); }

// tile_regs_commit (MATH side): hand the written DST section to PACK.
inline void math_dest_section_done() {
    math::set_math_semaphores();
    if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
        math_sync_tile_dst_index = 0;
        math::dest_section_flip();
    }
}

}  // namespace hw
}  // namespace compute
}  // namespace sst

#endif  // TRISC_MATH

#endif  // SST_COMPUTE_HW_MATH_H
