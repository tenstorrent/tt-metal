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
        TT_OP_UNPACR(SrcA, 0b00010001 /*CH0/CH1 Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb_zerosrc    = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID); // WA for tenstorrent/budabackend#1230
#endif

    const uint32_t outerloop     = narrow_tile ? 1 : 2;
    constexpr uint32_t innerloop = 1;

    if (unpack_to_dest)
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
        tmp.program(instrn_buffer);
    }
    else
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_zerosrc, unpack_srcb_set_dvalid);
        tmp.set_start_op(unpack_srca);
        tmp.program(instrn_buffer);
    }
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

    // In case of 32-bit integer numbers, we have to unpack into dest register
    const bool unpack_to_dest = (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt32)) ||
                                (unpack_src_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32));

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

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
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); // GPR preloaded with  16 | (16 << 16)

    _llk_unpack_tilize_mop_config_(narrow_tile, unpack_to_dest);
}

// Internal function to implement unpacking to source register
inline void unpack_tilize_impl(
    const std::uint32_t base_address, std::uint32_t num_loops, std::uint32_t top_face_offset_address, std::uint32_t bot_face_offset_address)
{
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    for (std::uint32_t n = 0; n < num_loops; n++)
    {
        std::uint32_t address = base_address + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

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

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Run MOP
        ckernel::ckernel_template::run(instrn_buffer);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

// Internal function to implement unpacking to destination register
inline void unpack_tilize_to_dest_impl(
    const std::uint32_t base_address,
    std::uint32_t unpack_src_format,
    std::uint32_t num_loops,
    std::uint32_t top_face_offset_address,
    std::uint32_t bot_face_offset_address)
{
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Unpack to dest register
    set_dst_write_addr(unp_cfg_context, unpack_src_format);
    wait_for_dest_available();

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);
    std::uint32_t address = base_address + top_face_offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

    // Get tile address
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Unpack top faces
    ckernel::ckernel_template::run(instrn_buffer);

    // Unpack bottom faces if needed
    if (num_loops > 1)
    {
        // Needed to stall counter reconfiguration until unpacker finishes previous instruction
        TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::UNPACK);

        // Don't clear the CH1 W counter - needed for multiple tiles
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1011);

        // Increment address to point to bottom faces in L1
        address += bot_face_offset_address;

        // Stall write to cfg until unpacker finishes
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

        // Get tile address
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Unpack bottom faces
        ckernel::ckernel_template::run(instrn_buffer);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context);
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
    std::uint32_t num_loops = narrow_tile ? 2 : num_faces / 2;

    if (!unpack_to_dest)
    {
        unpack_tilize_impl(base_address, num_loops, top_face_offset_address, bot_face_offset_address);
    }
    else
    {
        // Unpack tilize to DEST works with only one config context, hence it needs to be reset before calling the function.
        reset_config_context();
        unpack_tilize_to_dest_impl(base_address, unpack_src_format, num_loops, top_face_offset_address, bot_face_offset_address);
    }

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const bool narrow_tile = false, const std::uint32_t num_faces = 4)
{
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, (zero_srcA ? 0b010001 : 0b1), 0, 0, 0, 1, (zero_srcA ? 0 : 1), p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb = TT_OP_UNPACR(
        SrcB,
        (zero_srcA ? 0b010001 : (reload_srcB ? 0b0 : 0b1)),
        0,
        0,
        0,
        1,
        (zero_srcA ? 0 : 1),
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1);                                                                                         // Skip face ptr inc if same face is reloaded into srcB
    static constexpr uint unpack_neginf_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_NEGINFSRC); // Needed for max pool
    static constexpr uint unpack_zero_srca   = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);   // Needed for dot product
    static constexpr uint unpack_srcb_2_face = TT_OP_UNPACR(SrcB, 0b100010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product
    static constexpr uint unpack_srca_dat_valid = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);   // Needed for dot product
    static constexpr uint unpack_srcb_dat_valid =
        TT_OP_UNPACR(SrcB, (reload_srcB ? 0b0 : 0b1), 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot
                                                                                                                // product

    const uint32_t innerloop = zero_srcA ? (num_faces > 2 ? 2 : (num_faces - 1)) : 1;
    const uint32_t outerloop = zero_srcA ? 1 : (num_faces > 2) ? num_faces / 2 : num_faces;
    ckernel_template tmp(outerloop, innerloop, unpack_srca, ((zero_srcA && num_faces == 2) ? unpack_srcb_2_face : unpack_srcb));
    if constexpr (neginf_srcA)
    {
        tmp.set_start_op(unpack_neginf_srca);
    }
    else if constexpr (zero_srcA_reduce)
    {
        tmp.set_start_op(unpack_zero_srca);
    }
    else if constexpr (zero_srcA)
    {
        if (num_faces < 4)
        {
            tmp.set_start_op(unpack_zero_srca);
            tmp.set_end_ops(unpack_srca_dat_valid, unpack_srcb_dat_valid);
        }
    }
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
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * ((narrow_tile || (num_faces == 1)) ? FACE_C_DIM : TILE_C_DIM);

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, unpA_face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_face_r_dim * FACE_C_DIM - 1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32,
                 p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1,
        0,
        0,
        0,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::FACE_DIM_1x16); // GPR preloaded with  16 | (16 << 16)

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(narrow_tile, num_faces);
}

template <bool zero_srcA = false>
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
    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpA_src_format, tile_index_a) << (narrow_tile ? 0 : 1);

    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
    // For narrow tile we unpack 1 face in each iteration
    // Offset address is in 16B words
    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B   = block_ct_dim * ((narrow_tile || (num_faces == 1)) ? FACE_C_DIM / 16 : TILE_C_DIM / 16);
    std::uint32_t bot_face_offset_address = SCALE_DATUM_SIZE(unpA_src_format, face_r_dim * block_c_dim_16B); //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    std::uint32_t num_loops = narrow_tile ? 2 : ((num_faces > 1) ? num_faces / 2 : 1);

    // Clear z/w start counters for SrcB
    TTI_SETADCZW(UNP1, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    for (std::uint32_t n = 0; n < num_loops; n++)
    {
        std::uint32_t address_a = base_address_a + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        if constexpr (zero_srcA)
        {
            if (num_faces == 4 && n == 1)
            {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1011);
            }
            else
            {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
            }
        }
        else
        {
            TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
        }

        // Wait for free context
        wait_for_next_context(2);

        // Get tile address
        if (0 == unp_cfg_context)
        {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        }
        else
        {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
        }

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Run MOP
        if constexpr (zero_srcA)
        {
            if (num_faces == 4 && n == 0)
            {
                TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
            }

            ckernel::ckernel_template::run(instrn_buffer);

            if (num_faces == 4 && n != 0)
            {
                TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
            }
        }
        else
        {
            ckernel::ckernel_template::run(instrn_buffer);
        }

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM)
{
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);
    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32,
                 p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1,
        0,
        0,
        0,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::FACE_DIM_16x16); // GPR preloaded with  16 | (16 << 16)}
}
