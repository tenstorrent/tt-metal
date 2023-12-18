// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

inline void _llk_unpack_untilize_mop_config_() {
#if SKIP_UNP == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
    static constexpr uint unpack_addcr = TT_OP_NOP;
    static constexpr uint unpack_addr_offset = TT_OP_NOP;
    static constexpr uint unpack_wr_addr_offset = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b01000001, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_addcr = TT_OP_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001);
    static constexpr uint unpack_addr_offset =
        TT_OP_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
    static constexpr uint unpack_wr_addr_offset = TT_OP_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);
#endif

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_srca,
        unpack_srca,
        unpack_addr_offset,
        unpack_wr_addr_offset,
        0,
        unpack_addcr,
        TT_OP_NOP);

    tmp.program(instrn_buffer);
}

inline void _llk_unpack_untilize_hw_configure_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format) {
    configure_unpack_AB(
        unpack_src_format,
        unpack_src_format,
        unpack_dst_format,
        unpack_dst_format
    );
}

inline void _llk_unpack_untilize_init_(const std::uint32_t face_r_dim, std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size) {
    std::uint32_t unpA_ch1_x_stride = (uint) (unpack_dst_format&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    std::uint32_t unpA_ch1_y_stride = FACE_R_DIM*unpA_ch1_x_stride;

    TT_SETADCXX(p_setadc::UNP_A, face_r_dim*FACE_C_DIM-1, 0x0);

    unpack_tile_descriptor_u tile_descriptor;
    tile_descriptor.val[0] = 0;
    tile_descriptor.val[1] = 0;

    // Set descriptor 0
    tile_descriptor.f.in_data_format = unpack_src_format;
    tile_descriptor.f.uncompressed = 1;
    tile_descriptor.f.x_dim = FACE_C_DIM;

    // Set descriptor 1
    tile_descriptor.f.y_dim = FACE_R_DIM;
    tile_descriptor.f.z_dim = 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[1]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[1]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+1-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    std::uint32_t unpA_base_addr = (((unpack_dst_format & 0x3) == 1) ? 0x80 : 0x40)
        << UNP0_ADDR_BASE_REG_1_Base_SHAMT;  // base address skips halo rows in srcA (ch1)
    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_base_addr), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(unpA_base_addr), 0, HI_16(p_gpr_unpack::TMP0));

    std::uint32_t unpA_ch1_xy_stride = (unpA_ch1_x_stride << UNP0_ADDR_CTRL_XY_REG_1_Xstride_SHAMT) |
                                       (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_ch1_xy_stride), 0, LO_16(p_gpr_unpack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(unpA_ch1_xy_stride), 0, HI_16(p_gpr_unpack::TMP1));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, UNP0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_WRCFG(p_gpr_unpack::TMP1, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32);

    // Clear context state
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    unp_cfg_context = 0;

    const std::uint32_t tile_size_words = tile_size;
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_words), 0, LO_16(p_gpr_unpack::TILE_SIZE));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_size_words), 0, HI_16(p_gpr_unpack::TILE_SIZE));
    _llk_unpack_untilize_mop_config_();
}

template <bool first_pass = true>
inline void _llk_unpack_untilize_pass_(const std::uint32_t base_address, const std::uint32_t block_tile_cols) {
    std::uint32_t rem_blocks_in_row = block_tile_cols;

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b0010);  // Clear l1 addr y cnt
    if constexpr (first_pass) {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 0);
    } else {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 2);
    }

    // Wait for free context
    wait_for_next_context(1);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base_address;

    std::uint32_t face_2xr_cnt = 0;
    for (std::uint32_t r = 0; r < FACE_HEIGHT; r++) {
        rem_blocks_in_row = block_tile_cols;  // reset remaining blocks in row

        do {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2)) {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, 0);                                              // Run the MOP
#if SKIP_UNP == 1
                TTI_NOP;
#else
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // set data valid
#endif
                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            } else {
                TT_MOP(0, rem_blocks_in_row - 1, 0);  // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
                // if (face_2xr_cnt==FACE_HEIGHT/2) {
                //   TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); //set data valid
                //   TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                //   face_2xr_cnt = 0;
                //}
            }
        } while (rem_blocks_in_row > 0);

        TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));  // Clear offset pointer
        TTI_REG2FLOP(
            1,
            0,
            0,
            0,
            THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
            p_gpr::ZERO);                 // Clear offset register
        TTI_INCADCXY(0b001, 0, 0, 1, 0);  // inc l1 addr y cnt
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
