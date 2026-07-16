// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PACK-engine hardware configurations.
//
//   pack_hw_cfg               — one-shot format/stride programming (startup).
//   pack_dest_offset_cfg      — program the packer's DST-read offset registers.
//   pack_dest_cfg             — packer dest-state setup at startup (sync + reset
//                               + offset regs + addr counter).
//   pack_untilize_mop_cfg     — the untilize-mode PACR MOP (keyed on block width).
//   pack_untilize_row_cfg — per-row L1 strides + output offset (keyed on row width).
//   pack_untilize             — drain one DST block to L1 untilized (per block).
//   packer_wait_for_math_done / pack_dest_section_done — tile_regs_* (PACK).

#ifndef SST_COMPUTE_HW_PACK_H
#define SST_COMPUTE_HW_PACK_H

#include <cstdint>

#include "defs.h"

#ifdef TRISC_PACK

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cpack_common.h"
#include "llk_defs.h"

#include "experiments/static-state-tracking/inc/state.h"  // sst::TileConfig

namespace sst {
namespace compute {
namespace hw {

using namespace ckernel;
using namespace ckernel::packer;

// One-shot pack configure (dest read format == L1 write format, no conversion).
template <typename Traits>
inline void pack_hw_cfg(const sst::TileConfig& tile_config) {
    constexpr bool fp32 = Traits::fp32_dest_acc;
    const std::uint32_t df = tile_config.data_format;
    const std::uint32_t tile_size_bytes = sst::tensor::tile_size_bytes_from_tile_config(tile_config);
    configure_pack<fp32, ckernel::PackMode::Default>(
        df,
        df,
        tile_size_bytes,
        tile_config.face_r_dim,
        TILE_C_DIM,
        tile_config.num_faces,
        /*partial_face=*/false,
        /*relu_config=*/0);
}

template <typename Traits>
inline void pack_dest_offset_cfg() {
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    if constexpr (Traits::dst_sync == sst::DstSyncMode::SyncFull) {
        select_packer_dest_registers<DstSync::SyncFull>();
    } else {
        select_packer_dest_registers<DstSync::SyncHalf>();
    }
}

template <typename Traits>
inline void pack_dest_cfg() {
    tensix_sync();
    reset_dest_offset_id();
    pack_dest_offset_cfg<Traits>();
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

// --- Untilize PACK configure, split into two independent sub-steps so the state
// tracker can reprogram only what actually changed (granular reconfiguration).
//
//   pack_untilize_mop_cfg<TilesPerBlock>     — the strided-untilize PACR MOP + addr modifiers +
//                                within-face setup. Its shape is a function of the
//                                block width (MOP inner loop = TilesPerBlock), face_r_dim
//                                (MOP outer loop) and the op mode. This is the
//                                EXPENSIVE piece (a full ckernel_template::program()
//                                plus a replay-buffer load).
//   pack_untilize_row_cfg<TilesPerRow> — the per-row L1 Z-stride and the output-row address
//                                offset. A function of the data format, face_r_dim
//                                and the FULL row width (TilesPerRow) only — independent of
//                                the block width. This is the CHEAP tail (one cfg
//                                RMW + two SETDMAREG + a WRCFG).
//
// Splitting them means a kernel that changes only the block width reprograms the
// MOP but keeps the strides; a kernel that changes only the row width keeps the
// MOP and reprograms just the strides.
template <std::uint32_t TilesPerBlock>
inline void pack_untilize_mop_cfg(const sst::TileConfig& tile_config) {
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 0}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 1, .clr = 0}}.set(ADDR_MOD_1);

    const std::uint32_t MOP_INNER_LOOP = TilesPerBlock;
    const std::uint32_t MOP_OUTER_LOOP = tile_config.face_r_dim;
    const std::uint32_t PACK_INTF_SEL = p_pacr::TWO_INTFS_ACTIVE;

    ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),  // w cnt -> next tile
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010 /*CH0_W*/));  // W = W_Cr

    constexpr std::uint32_t replay_buf_len = 2;
    load_replay_buf(ckernel::packer::replay_buf_offset, replay_buf_len, [] {
        // THCON_SEC0_REG1_L1_Dest_addr += SCRATCH_SEC2 (per-row L1 stride)
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_NOP;
    });
    tmp.set_end_op(lltt::replay_insn(ckernel::packer::replay_buf_offset, replay_buf_len));

    const std::uint32_t last_loop_op = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        PACK_INTF_SEL,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    tmp.set_last_inner_loop_instr(last_loop_op);
    tmp.set_last_outer_loop_instr(last_loop_op);
    tmp.program();

    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

template <std::uint32_t TilesPerRow>
inline void pack_untilize_row_cfg(const sst::TileConfig& tile_config) {
    const std::uint32_t df = tile_config.data_format;

    const std::uint32_t x_stride = (df & 0x3) == to_underlying(DataFormat::Float32)   ? 4
                                   : (df & 0x3) == to_underlying(DataFormat::Float16) ? 2
                                                                                      : 1;
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = 2 * tile_config.face_r_dim * y_stride;
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(z_stride);

    const std::uint32_t output_addr_offset =
        SCALE_DATUM_SIZE(df, TilesPerRow * ((tile_config.num_faces == 1) ? 1 : 2) * FACE_C_DIM);
    TT_SETDMAREG(0, LOWER_HALFWORD(output_addr_offset / 16), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(output_addr_offset / 16), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR_OFFSET, 0, SCRATCH_SEC2_val_ADDR32);
    TTI_NOP;
}

// Drain one block (TilesPerBlock tiles) from DST to L1 untilized. `out_l1_addr_16B`
// is the reserved output CB write pointer; `col_tile_offset` is the number of
// output-row tiles to the LEFT of this block (0 for the first block in a row).
// Taking an explicit tile offset (rather than block_index * uniform_width) lets a
// row be packed as a sequence of DIFFERENT-width blocks and still land each block
// at the correct column — which is what the granularity benchmark needs.
template <std::uint32_t TilesPerBlock>
inline void pack_untilize(
    std::uint32_t out_l1_addr_16B,
    std::uint32_t col_tile_offset,
    const sst::TileConfig& tile_config,
    std::uint32_t dst_tile_index = 0) {
    const std::uint32_t col_faces = (tile_config.num_faces > 2) ? (tile_config.num_faces / 2) : tile_config.num_faces;
    const std::uint32_t off16 =
        SCALE_DATUM_SIZE(tile_config.data_format, col_tile_offset * col_faces * FACE_C_DIM) / 16;
    const std::uint32_t address = out_l1_addr_16B - 1 + off16;

    program_packer_destination(address);

    const std::uint32_t num_faces_per_rdim_tile = (tile_config.num_faces > 2) ? 2 : 1;
    const std::uint32_t tile_dst_offset = dst_tile_index;

    TTI_SETADCZW(p_setadc::PAC, 0 /*Ch1_W*/, 0 /*Ch1_Z*/, 0 /*Ch0_W*/, 0 /*Ch0_Z*/, 0b0001 /*write Ch0_Z*/);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, (15 + tile_dst_offset) & 0xF);
    TTI_SETADCXY(p_setadc::PAC, 0 /*Ch1_Y*/, 0 /*Ch1_X*/, 0 /*Ch0_Y*/, 0 /*Ch0_X*/, 0b0011 /*write Ch0_X+Ch0_Y*/);

    for (std::uint32_t face = 0; face < num_faces_per_rdim_tile; face++) {
        ckernel::ckernel_template::run();
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        TTI_SETADCXY(p_setadc::PAC, 0 /*Ch1_Y*/, 0 /*Ch1_X*/, 0 /*Ch0_Y*/, 0 /*Ch0_X*/, 0b0010 /*write Ch0_Y*/);
    }

    TTI_SETADCZW(p_setadc::PAC, 0 /*Ch1_W*/, 0 /*Ch1_Z*/, 0 /*Ch0_W*/, 0 /*Ch0_Z*/, 0b0101 /*write Ch0_Z+Ch1_Z*/);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_dst_offset);
}

// tile_regs_wait (PACK side): block until MATH signals the DST section is ready.
inline void packer_wait_for_math_done() {
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

// tile_regs_release (PACK side): free the drained DST section back to MATH.
template <typename Traits>
inline void pack_dest_section_done() {
    constexpr bool fp32 = Traits::fp32_dest_acc;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    if constexpr (Traits::dst_sync == sst::DstSyncMode::SyncFull) {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, fp32, 0, ADDR_MOD_1, 0);
    } else {
        TT_ZEROACC(p_zeroacc::CLR_HALF, fp32, 0, ADDR_MOD_1, dest_offset_id % 2);
    }
    t6_semaphore_get<p_stall::NONE>(semaphore::MATH_PACK);
    if constexpr (Traits::dst_sync == sst::DstSyncMode::SyncHalf) {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<DstSync::SyncHalf>();
    }
}

}  // namespace hw
}  // namespace compute
}  // namespace sst

#endif  // TRISC_PACK

#endif  // SST_COMPUTE_HW_PACK_H
