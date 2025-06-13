// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

#ifndef SKIP_UNP
#define SKIP_UNP (0)
#endif

inline void _llk_unpack_untilize_mop_config_()
{
    constexpr uint replay_buf_len = (SKIP_UNP == 1) ? 1 : 6;
    load_replay_buf(
        0,
        replay_buf_len,
        // Lambda function to set up replay buffer
        []
        {
#if SKIP_UNP == 1
            TTI_NOP;
#else
            TTI_DMANOP; // WRCFG that sets offset in previous loop needs additional cycle to complete
            TTI_UNPACR(SrcA, 0b01000001 /*CH1_Y+=1, CH0_Z+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b01000001 /*CH1_Y+=1, CH0_Z+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
            // Need to stall WRCFG on the addition from ADDDMAREG
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            // Resets SrcA Z counter CR, which should point to initial Z counter value
            TTI_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001 /*CH0_Z*/);
#endif
        });

#if SKIP_UNP == 1
    static constexpr uint load_offset_addr_cntx0 = TT_OP_NOP;
    static constexpr uint load_offset_addr_cntx1 = TT_OP_NOP;
#else
    static constexpr uint load_offset_addr_cntx0 = TT_OP_WRCFG(p_gpr_unpack::TILE_OFFSET, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_address_ADDR32);
    static constexpr uint load_offset_addr_cntx1 = TT_OP_WRCFG(p_gpr_unpack::TILE_OFFSET, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32);
#endif

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        false, // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_len),
        0,
        0,
        0,
        lltt::replay_insn(0, replay_buf_len),
        load_offset_addr_cntx0,
        load_offset_addr_cntx1);
    tmp.program(instrn_buffer);
}

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void _llk_unpack_untilize_hw_configure_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t num_faces                   = 4)
{
    constexpr bool is_row_pool  = false;
    constexpr bool stoch_rnd_en = (stoch_rnd_mode == StochRndType::All);
    constexpr bool fpu_srnd_en  = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Pack);
    configure_unpack_AB<is_fp32_dest_acc_en, is_row_pool, fpu_srnd_en, pack_srnd_en>(
        unpack_src_format, unpack_src_format, unpack_dst_format, unpack_dst_format, face_r_dim, face_r_dim, within_face_16x16_transpose, num_faces, num_faces);
}

inline void _llk_unpack_untilize_init_(
    const std::uint32_t unpack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const std::uint32_t unpA_ch1_x_stride = (unpack_dst_format & 0x3) == (std::uint32_t)DataFormat::Float32   ? 4
                                            : (unpack_dst_format & 0x3) == (std::uint32_t)DataFormat::Float16 ? 2
                                                                                                              : 1;
    const std::uint32_t unpA_ch1_y_stride = FACE_R_DIM * unpA_ch1_x_stride;

    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Get pointer to registers for current state ID
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xFFFF>(FACE_C_DIM);
    // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 <<
    // 16)
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_1x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32); // GPR preloaded with  16 | (16 << 16)

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_size), 0, HI_16(p_gpr_unpack::TILE_SIZE));

    _llk_unpack_untilize_mop_config_();
}

template <bool first_pass = true>
inline void _llk_unpack_untilize_pass_(const std::uint32_t base_address, const std::uint32_t block_tile_cols)
{
    std::uint32_t rem_blocks_in_row = block_tile_cols;

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b0010); // Clear l1 addr y cnt
    if constexpr (first_pass)
    {
        // Select top faces in the 1st pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 0);
    }
    else
    {
        // Select bottom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 2);
    }

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base_address;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = base_address;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    std::uint32_t face_2xr_cnt = 0;
    for (std::uint32_t r = 0; r < FACE_HEIGHT; r++)
    {
        rem_blocks_in_row = block_tile_cols; // reset remaining blocks in row

        do
        {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2))
            {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, unp_cfg_context == 0 ? 0 : 0xff); // Run the MOP
#if SKIP_UNP == 1
                TTI_NOP;
#else
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // set data valid
                TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
#endif
                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            }
            else
            {
                TT_MOP(0, rem_blocks_in_row - 1, unp_cfg_context == 0 ? 0 : 0xff); // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
                // if (face_2xr_cnt==FACE_HEIGHT/2) {
                //   TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); //set data valid
                //   TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                //   face_2xr_cnt = 0;
                //}
            }
        } while (rem_blocks_in_row > 0);

        TTI_MULDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr::ZERO); // TILE_OFFSET=TILE_OFFSET*0
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        if (0 == unp_cfg_context)
        {
            TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_address_ADDR32);
        }
        else
        {
            TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32);
        }
        TTI_INCADCXY(0b001, 0, 0, 1, 0); // inc l1 addr y cnt
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
