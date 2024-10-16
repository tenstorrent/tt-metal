// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void _llk_unpack_tilize_mop_config_(const bool narrow_tile=false) {
    #if SKIP_UNP == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
        static constexpr uint unpack_srcb_zerosrc = TT_OP_NOP;
        static constexpr uint unpack_srcb_set_dvalid = TT_OP_NOP;
    #else
        static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    #endif

    const uint32_t outerloop = 1;
    constexpr uint32_t innerloop = 1;
    ckernel_template tmp(outerloop, innerloop, unpack_srcb_set_dvalid);
    tmp.set_start_op(unpack_srca);
    tmp.program(instrn_buffer);
}

template <bool is_fp32_dest_acc_en = false, StochRndType stoch_rnd_mode = StochRndType::None>
inline void _llk_unpack_tilize_hw_configure_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM,  const std::uint32_t within_face_16x16_transpose = 0, const std::uint32_t num_faces = 4) {

    constexpr bool is_row_pool = false;
    constexpr bool stoch_rnd_en = (stoch_rnd_mode == StochRndType::All);
    constexpr bool fpu_srnd_en = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = stoch_rnd_en ||(stoch_rnd_mode == StochRndType::Pack);

    configure_unpack_AB<is_row_pool, is_fp32_dest_acc_en, fpu_srnd_en, pack_srnd_en>(
        unpack_src_format,
        unpack_src_format,
        unpack_dst_format,
        unpack_dst_format,
        face_r_dim,
        face_r_dim,
        within_face_16x16_transpose,
        num_faces,
        num_faces);
}

inline void _llk_unpack_tilize_init_(const std::uint32_t unpack_src_format=0, const std::uint32_t unpack_dst_format=0, const std::uint32_t ct_dim=0, const std::uint32_t face_r_dim=FACE_R_DIM, const bool narrow_tile=false) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim*FACE_C_DIM-1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

    // below is the configuration for 64-row unpack for srca
    const uint Tile_x_dim = 1024;
    const uint Tile_z_dim = 1;
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));
    //Force x-dim to 1024
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0xffff0000>(0 | (Tile_x_dim<<16));
    //Force z-dim to CNT_ZDIM/4
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 0, 0xffff0000>(0 | (Tile_z_dim<<16));

    //Force x-end for Unpackers to 1024
    TTI_SETADCXX(p_setadc::UNP0, 1023, 0x0);

    _llk_unpack_tilize_mop_config_(narrow_tile);
}

inline void _llk_unpack_tilize_(const std::uint32_t base_address, const std::uint32_t tile_index, std::uint32_t unpack_src_format=0, std::uint32_t block_ct_dim=0, const std::uint32_t face_r_dim=FACE_R_DIM, const std::uint32_t num_faces=4, const bool narrow_tile=false) {
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID


    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);
                                                    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
                                                    // For narrow tile we unpack 1 face in each iteration
                                                    // Offset address is in 16B words
                                                    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B = block_ct_dim * (narrow_tile ? FACE_C_DIM/16 : TILE_C_DIM/16);
    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE(unpack_src_format, face_r_dim*block_c_dim_16B);  //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    // FIXME MT: This should be revisited for narrow tiles
    // std::uint32_t num_loops = narrow_tile ? 2 : num_faces/2;

    std::uint32_t address = base_address + top_face_offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

    // Wait for free context
    wait_for_next_context(2);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    }

    // Run MOP
    ckernel::ckernel_template::run(instrn_buffer);

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
