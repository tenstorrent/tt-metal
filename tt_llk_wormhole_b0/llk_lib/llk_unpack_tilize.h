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
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*CH0/CH1 Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb_zerosrc    = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID); // WA for tenstorrent/budabackend#1230

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

/*************************************************************************
 * LLK UNPACK FAST TILIZE (Tilize single input using both unpackers and packer)
 * full_dim is the tensor width in number of tiles
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim is 1 (only if full_dim is 1) or 2 and 3 (for any other full_dim)
 * each call can process unit_dim * num_units tiles but when unit_dim is 2 or 3 all tiles must be in a single row
 * changing between unit_dim 1 and 2/3 requires reconfiguration while changing between 2 and 3 does not
 * base_address is the 16B base address of the start of the tile row
 * tile_index is the index of the tile inside that row
 * currently supports only 4 16x16 faces per tile
 * supported input formats are: FP32 (via FP16 or TF32) or FP16_B
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_hw_configure_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    configure_unpack_AB<is_fp32_dest_acc_en>(unpack_src_format, unpack_src_format, unpack_dst_format, unpack_dst_format);
}

inline void _llk_unpack_fast_tilize_mop_config_()
{
    // Y moves to the next tile, Z moves to the next row (both ch0 and ch1)
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1 = 0b00'10'00'01;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0 = 0b00'00'10'00;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1 = 0b00'11'00'01;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0 = 0b00'00'11'00;

    // UNPACR instructions are used with unit_dim 2 and SKIP instructions are used with unit_dim 3
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,
        false,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0),
        TT_OP_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0),
        TT_OP_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0));

    tmp.program(instrn_buffer);
}

inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, std::uint32_t full_dim)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // save the following state that is going to be modified:
    // tile x, y, and z dims for both unpackers
    // CH1 Z stride for both unpackers
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // set x dim to single tile width, moving across y counter moves to the next tile in row major
    // set y dim to full dim, moving across z counter moves to the next row in row major
    // set z dim to single face height, moving across w counter moves to the next face row in row major
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, full_dim, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, FACE_R_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, full_dim, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, FACE_R_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // for unit_dim 2 or 3 unpacker read sizes are multiples of 32 datums (64 or 96) so CH1 Z stride is set to 32 datums
    // for unit_dim 1 unpacker reads whole tile per iteration so CH1 counter is not used
    // why are CH1 strides in bytes?
    // SCALE_DATUM_SIZE wouldn't work here since it doesn't have a case for TF32
    uint ch1_x_stride = (uint)(unpack_dst_format & 0x3) == (uint)DataFormat::Float32 ? 4 : 2;
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(TILE_C_DIM * ch1_x_stride);
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_uninit_()
{
    // restore saved state
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, p_cfg::WRCFG_32b, UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // reset all counters
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
}

inline void _llk_unpack_fast_tilize_block_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units,
    const std::uint32_t full_dim)
{
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    uint32_t address = base_address + (SCALE_DATUM_SIZE(unpack_src_format, tile_index * TILE_C_DIM) >> 4); // move by tile width in 16B words
    // for unit_dim 2 UNPA reads top faces and UNPB reads bottom faces
    // for unit_dim 3 UNPA reads top 8 rows of top then bottom faces, UNPB reads bottom 8 rows of top then bottom faces
    uint32_t unpB_row_offset = unit_dim == 2 ? FACE_R_DIM : (FACE_R_DIM / 2);
    uint32_t unpB_address    = address + (SCALE_DATUM_SIZE(unpack_src_format, full_dim * TILE_C_DIM * unpB_row_offset) >> 4);

    // reset all counters since X start and end are set after this
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));

    // unit_dim 1 reads the whole tile while unit_dim 2 and 3 read one row from 2 or 3 tiles
    if (unit_dim == 1)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, TILE_R_DIM * TILE_C_DIM - 1, 0x0);
    }
    else if (unit_dim == 2)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, 2 * TILE_C_DIM - 1, 0x0);
    }
    else if (unit_dim == 3)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, 3 * TILE_C_DIM - 1, 0x0);
    }
    else
    {
        // replace this with a proper assert once it's available
        // FWASSERT("Unsupported unit_dim", false);
    }

    wait_for_next_context(2);

    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = unpB_address;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = unpB_address;
    }

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Y moves to the next tile, Z moves to the next row (both ch0 and ch1)
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1 = 0b00'10'00'01;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0 = 0b00'00'10'00;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1 = 0b00'11'00'01;
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0 = 0b00'00'11'00;

    for (std::uint32_t i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // read whole tile contiguously then move two face rows down to the next tile
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 1);
            TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 2, 0);
        }
        else if (unit_dim == 2)
        {
            // read top(A)/bottom(B) faces of two tiles in a row (4 faces each), switch bank,
            // then move to the next two tiles (CH0Y += 2) and back to the top of a tile (CH01Z = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < FACE_R_DIM - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        }
        else if (unit_dim == 3)
        {
            // read top 8(A)/bottom 8(B) rows of top faces of three tiles in a row (6 halves of a face each), switch bank,
            // then move to the bottom faces (CH0W = 1) and back to the top of a face (CH01Z = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < (FACE_R_DIM / 2) - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, ((FACE_R_DIM / 2) - 1) - 1, 0xFFFF);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 1, 0, SETADC_CH01(p_setadc::ZW));

            // read top 8(A)/bottom 8(B) rows of bottom faces of three tiles in a row (6 halves of a face each), switch bank,
            // then move to the top faces of the next three tiles (CH0Y += 3) and back to top of a tile (CH01Z = 0, CH0W = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < (FACE_R_DIM / 2) - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, ((FACE_R_DIM / 2) - 1) - 1, 0xFFFF);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        }
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    switch_config_context(unp_cfg_context);
}
