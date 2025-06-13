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

inline void _llk_unpack_tilize_mop_config_(const bool narrow_tile = false, const bool unpack_to_dest = false)
{
#if SKIP_UNP == 1
    static constexpr uint unpack_srca            = TT_OP_NOP;
    static constexpr uint unpack_srca_to_dest    = TT_OP_NOP;
    static constexpr uint unpack_srcb_zerosrc    = TT_OP_NOP;
    static constexpr uint unpack_srcb_set_dvalid = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca_to_dest =
        TT_OP_UNPACR(0, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
#endif
    const uint32_t outerloop = 1;
    const uint32_t innerloop = 1;

    ckernel_template tmp(outerloop, innerloop, unpack_to_dest ? unpack_srca_to_dest : unpack_srcb_set_dvalid);

    if (!unpack_to_dest)
    {
        tmp.set_start_op(unpack_srca);
    }

    tmp.program(instrn_buffer);
}

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void _llk_unpack_tilize_hw_configure_(
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

inline void _llk_unpack_tilize_init_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

    // In case of 32-bit integer numbers, we have to unpack into dest register
    const bool unpack_to_dest = (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt32)) ||
                                (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32));

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 <<
    // 16)

    // below is the configuration for 64-row unpack for srca
    const uint Tile_x_dim = 1024;
    const uint Tile_z_dim = 1;
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));
    // Force x-dim to 1024
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0xffff0000>(0 | (Tile_x_dim << 16));
    // Force z-dim to 1 as X dim is set to cover the entire tile, so no need to iterate over faces.
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(0 | (Tile_z_dim << 16));

    // Force x-end for Unpackers to 1024
    TTI_SETADCXX(p_setadc::UNP0, 1023, 0x0);

    _llk_unpack_tilize_mop_config_(narrow_tile, unpack_to_dest);
}

inline void _llk_unpack_tilize_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    std::uint32_t unpack_src_format = 0,
    std::uint32_t block_ct_dim      = 0,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool narrow_tile          = false)
{
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // In case of 32-bit integer numbers, we have to unpack into dest register
    const bool unpack_to_dest = (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt32)) ||
                                (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32));

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);
    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
    // For narrow tile we unpack 1 face in each iteration
    // Offset address is in 16B words
    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B   = block_ct_dim * (narrow_tile ? FACE_C_DIM / 16 : TILE_C_DIM / 16);
    std::uint32_t bot_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, face_r_dim * block_c_dim_16B); //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    // FIXME MT: This should be revisited for narrow tiles
    // std::uint32_t num_loops = narrow_tile ? 2 : num_faces/2;

    std::uint32_t address = base_address + top_face_offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    if (unpack_to_dest)
    {
        // Unpack to dest
        set_dst_write_addr(unp_cfg_context, unpack_src_format);
        wait_for_dest_available();
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run(instrn_buffer);

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (unpack_to_dest)
    {
        unpack_to_dest_tile_done(unp_cfg_context);
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}

inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const std::uint32_t num_faces = 4, const std::uint32_t face_r_dim = FACE_R_DIM)
{
    // Revert X dim value to default.
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Revert Z dim value back to default.
    const uint Tile_z_dim = num_faces;
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xffff0000>(Tile_z_dim);

    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // GPR preloaded with  16 | (16 << 16)}
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

// TODO: add support for all the template parameters
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const bool narrow_tile = false, const std::uint32_t num_faces = 4)
{
    const std::uint32_t replay_buf_run_len  = 6;
    const std::uint32_t replay_buf_half_len = replay_buf_run_len >> 1;

    // Lambda function to set up replay buffer
    load_replay_buf(
        0,
        replay_buf_run_len,
        []
        {
            // Unpacks 1x16 row of datums to SrcA
            TTI_UNPACR(SrcA, 0b01000000 /*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

            // THCON_SEC0_REG3_Base_address_ADDR32 =  THCON_SEC0_REG3_Base_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;

            // Unpacks 1x16 row of datums to SrcA
            TTI_UNPACR(SrcA, 0b01000000 /*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

            // THCON_SEC0_REG3_Base_cntx1_address_ADDR32 =  THCON_SEC0_REG3_Base_cntx1_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        });

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                     // src B
        false,                                     // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_half_len), // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_half_len, replay_buf_half_len), // runs when context is 1
        0,
        0);

    tmp.program(instrn_buffer);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const bool narrow_tile,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces       = 4,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM)
{
    // Sets the block_c_dim for unpack to use to increment the L1 address
    const std::uint32_t c_dim_size = SCALE_DATUM_SIZE(unpack_src_format, ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM)) >> 4;

    // This sets the scartch register that CFGSHIFTMASK instruction uses to increment the L1 address
    TT_SETDMAREG(0, LOWER_HALFWORD(c_dim_size), 0, LO_16(p_gpr_unpack::TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(c_dim_size), 0, HI_16(p_gpr_unpack::TILE_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TILE_OFFSET, 0, SCRATCH_SEC0_val_ADDR32);
    TTI_NOP;

    // Unpack 1 row of 1x16 at a time for SrcA
    config_unpacker_x_end<p_setadc::UNP_A>(1);
    config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);

    // Set Y stride for SrcA to be one 1x16 row of datums
    uint unpA_ch1_y_stride = SCALE_DATUM_SIZE(unpack_dst_format, FACE_C_DIM);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(narrow_tile, num_faces);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_(
    std::uint32_t unpA_src_format,
    std::uint32_t face_r_dim,
    std::uint32_t narrow_tile,
    std::uint32_t base_address_a,
    std::uint32_t address_b,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4)
{
    const std::uint32_t offset_address_a = SCALE_DATUM_SIZE(unpA_src_format, tile_index_a) << 1;
    const std::uint32_t address_a        = base_address_a + offset_address_a;

    const std::uint32_t block_c_dim = block_ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM) * face_r_dim;
    const bool run_r_dim_loop       = (face_r_dim > 1);

    volatile uint tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Clear z/w start counters for SrcA/B
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);

    for (std::uint32_t n = 0; n < num_faces; n++)
    {
        /*
        Face 0: address = base_address
        Face 1: address = base_address + 1x16 row of datums
        Face 2: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim (address for the bottom 2 faces of tiles)
        Face 3: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim + 1x16 row of datums
        */
        std::uint32_t address_face_a = (n % 2 == 0) ? address_a : (address_a + (SCALE_DATUM_SIZE(unpA_src_format, FACE_C_DIM) >> 4));
        address_face_a += (n >= 2) ? ((SCALE_DATUM_SIZE(unpA_src_format, block_c_dim)) >> 4) : 0;

        // Wait for free context
        wait_for_next_context(2);

        if constexpr (neginf_srcA)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC);
        }
        else if constexpr (zero_srcA_reduce)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
        }

        // Get tile address
        if (0 == unp_cfg_context)
        {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_face_a;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        }
        else
        {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_face_a;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
        }

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Reset Y counters for SrcA
        TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010);
        // Unpack SrcB 16x16 face & Set Data Valid

        // If reload_srcB, only first face needs to be loaded, otherwise CH0_Z+=1
        TTI_UNPACR(SrcB, reload_srcB ? 0b0 : 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // Unpacks face_r_dim-1 rows of 1x16 datums to SrcA
        if (run_r_dim_loop)
        {
            ckernel_unpack_template::run(instrn_buffer, face_r_dim - 1, unp_cfg_context == 0 ? 0 : 0xffff);
        }

        // Unpack last SrcA row of a 16x16 face and SetDvalid
        TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}
