// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {
    constexpr bool within_face_16x16_transpose = false;
    constexpr StochRndType stoch_rnd_mode = StochRndType::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {.unpA_operand = unpA_operand};
    llk_unpack_tilize_hw_configure<is_fp32_dest_acc_en>(&unpack_tilize_params);
}

inline void llk_unpack_tilize_mop_config(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    _llk_unpack_tilize_mop_config_(narrow_tile);
}

inline void llk_unpack_tilize_init(const std::uint32_t operand, const std::uint32_t ct_dim) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    _llk_unpack_tilize_init_(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], ct_dim, face_r_dim, narrow_tile);
}

inline void llk_unpack_tilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);
    std::uint32_t operand_id = get_operand_id(operand);
    unpack_config_u config = {0};

    config.f.out_data_format = (uint)unpack_dst_format[operand_id];
    config.f.throttle_mode = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0,0,THCON_SEC0_REG2_Out_data_format_ADDR32);
    // GPR preloaded with  16 | (16 << 16)}
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;
}

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor

    WAYPOINT("UPTW");
    _llk_unpack_tilize_(
        base_address, tile_index, unpack_src_format[operand_id], block_ct_dim, face_r_dim, num_faces, narrow_tile);
    WAYPOINT("UPTD");
}

inline void llk_unpack_tilize_block(std::uint32_t operand, std::uint32_t block_c_tiles) {
    for (std::uint32_t tile_index = 0; tile_index < block_c_tiles; tile_index++) {
        llk_unpack_tilize(operand, tile_index, block_c_tiles);
    }
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

template <bool is_fp32_dest_acc_en = false, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_tilizeA_B_hw_configure(
    const llk_unpack_AB_params_t *unpack_tilizeA_B_params, const int within_face_16x16_transpose = 0) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpack_tilizeA_B_params->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_tilizeA_B_params->unpB_operand);

    // unpA -> srcA
    // Unpack only 1x16 row of datums to SrcA per UNPACK instruction
    const uint32_t num_faces_a = get_operand_num_faces(unpA_operand_id);
    const uint32_t face_r_dim_a = get_operand_face_r_dim(unpA_operand_id);

    // unpB -> srcB
    const uint32_t num_faces_b = get_operand_num_faces(unpB_operand_id);
    const uint32_t face_r_dim_b = get_operand_face_r_dim(unpB_operand_id);
    configure_unpack_AB<false, is_fp32_dest_acc_en, false, false>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        face_r_dim_a,
        face_r_dim_b,
        within_face_16x16_transpose,
        num_faces_a,
        num_faces_b);
}

template <bool is_fp32_dest_acc_en = false, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_tilizeA_B_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_AB_params_t unpack_tilizeA_B_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_tilizeA_B_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(
        &unpack_tilizeA_B_params, within_face_16x16_transpose);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false>
inline void llk_unpack_tilizeA_B_mop_config(const bool narrow_tile = false, const std::uint32_t num_faces = 4) {

    const std::uint32_t replay_buf_run_len = 6;
    const std::uint32_t replay_buf_half_len = replay_buf_run_len >> 1;

    // Lambda function to set up replay buffer
    load_replay_buf(0, replay_buf_run_len, false, []{
        //Unpacks 1x16 row of datums to SrcA
        TTI_UNPACR(SrcA, 0b01000000/*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // THCON_SEC0_REG3_Base_address_ADDR32 =  THCON_SEC0_REG3_Base_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;

        //Unpacks 1x16 row of datums to SrcA
        TTI_UNPACR(SrcA, 0b01000000/*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // THCON_SEC0_REG3_Base_cntx1_address_ADDR32 =  THCON_SEC0_REG3_Base_cntx1_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
    });

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,  // src B
        false,  // halo - just used for 4 unpacks
        TT_OP_REPLAY(0, replay_buf_half_len, 0, 0), // runs when context is 0
        0,
        0,
        0,
        TT_OP_REPLAY(replay_buf_half_len, replay_buf_half_len, 0, 0), // runs when context is 1
        0,
        0);

    tmp.program(instrn_buffer);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces = 4,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM) {

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    //Sets the block_c_dim for unpack to use to increment the L1 address
    const std::uint32_t c_dim_size = SCALE_DATUM_SIZE(unpack_src_format[operandA_id], ct_dim * ((num_faces==1) ? FACE_C_DIM: TILE_C_DIM)) >> 4;

    //This sets the scartch register that CFGSHIFTMASK instruction uses to increment the L1 address
    TT_SETDMAREG(0, LOWER_HALFWORD(c_dim_size), 0, LO_16(p_gpr_unpack::TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(c_dim_size), 0, HI_16(p_gpr_unpack::TILE_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TILE_OFFSET, 0, SCRATCH_SEC0_val_ADDR32);
    TTI_NOP;

    //Unpack 1 row of 1x16 at a time for SrcA
    config_unpacker_x_end<p_setadc::UNP_A>(1);
    config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);

    //Set Y stride for SrcA to be one 1x16 row of datums
    uint unpA_ch1_y_stride = SCALE_DATUM_SIZE(unpack_dst_format[operandA_id], FACE_C_DIM);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    llk_unpack_tilizeA_B_mop_config(narrow_tile, num_faces);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4,
    std::uint32_t unpA_face_r_dim = FACE_R_DIM) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);

    //TODO: RT face_r_dim should be taken from get_operand_face_r_dim(operandA_id);
    //But currently ops do not populate that array correctly
    const std::uint32_t face_r_dim = unpA_face_r_dim;

    const std::uint32_t base_address_a = cb_interface[operandA_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    const std::uint32_t offset_address_a = SCALE_DATUM_SIZE(unpack_src_format[operandA_id], tile_index_a) << 1;
    const std::uint32_t address_a = base_address_a + offset_address_a;

    const std::uint32_t base_address_b = cb_interface[operandB_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    const std::uint32_t offset_address_b = tile_index_b * cb_interface[operandB_id].fifo_page_size;
    const std::uint32_t address_b = base_address_b + offset_address_b;

    const std::uint32_t block_c_dim = block_ct_dim * ((num_faces==1) ? FACE_C_DIM: TILE_C_DIM) * face_r_dim;
    const bool run_r_dim_loop = (face_r_dim > 1);

    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Clear z/w start counters for SrcA/B
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);

    WAYPOINT("UPTW");

    for (std::uint32_t n = 0; n < num_faces; n++) {

        /*
        Face 0: address = base_address
        Face 1: address = base_address + 1x16 row of datums
        Face 2: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim (address for the bottom 2 faces of tiles)
        Face 3: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim + 1x16 row of datums
        */
        std::uint32_t address_face_a = (n % 2 == 0) ? address_a : (address_a + (SCALE_DATUM_SIZE(unpack_src_format[operandA_id], FACE_C_DIM) >> 4));
        address_face_a += (n >= 2) ? ((SCALE_DATUM_SIZE(unpack_src_format[operandA_id], block_c_dim)) >> 4) : 0;

        // Wait for free context
        wait_for_next_context(2);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        if constexpr (neginf_srcA) {
            TTI_UNPACR_NOP(SrcA,0,0,0,0,0,0,p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC);
        }

        // Get tile address
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_face_a;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_face_a;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
        }

        //Reset Y counters for SrcA
        TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010);
        //Unpack SrcB 16x16 face & Set Data Valid

        //If reload_srcB, only first face needs to be loaded, otherwise CH0_Z+=1
        TTI_UNPACR(SrcB, reload_srcB ? 0b0 : 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        //Unpacks face_r_dim-1 rows of 1x16 datums to SrcA
        if (run_r_dim_loop) {
            ckernel_unpack_template::run(instrn_buffer, face_r_dim-1, unp_cfg_context == 0 ? 0 : 0xffff);
        }

        //Unpack last SrcA row of a 16x16 face and SetDvalid
        TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }

    WAYPOINT("UPTD");
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false>
inline void llk_unpack_tilizeA_B_block(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t block_c_tiles_a,
    std::uint32_t tile_idx_b,
    std::uint32_t num_faces = 4,
    std::uint32_t unpA_face_r_dim = FACE_R_DIM) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<neginf_srcA, reload_srcB, zero_srcA>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces, unpA_face_r_dim);
    }
}
