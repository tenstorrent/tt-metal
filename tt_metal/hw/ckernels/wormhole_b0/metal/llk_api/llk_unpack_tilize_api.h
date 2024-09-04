// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK TILIZE
*************************************************************************/

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {

    constexpr bool  within_face_16x16_transpose = false;
    constexpr StochRndType stoch_rnd_mode = StochRndType::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces
    );
}


template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure_disaggregated(
    const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand
    };
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
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        ct_dim,
        face_r_dim,
        narrow_tile
    );

}

inline void llk_unpack_tilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim*FACE_C_DIM-1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim*FACE_C_DIM-1, 0x0);
    std::uint32_t operand_id = get_operand_id(operand);
    unpack_config_u config = {0};

    config.f.out_data_format = (uint)unpack_dst_format[operand_id];
    config.f.throttle_mode = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16); //GPR preloaded with  16 | (16 << 16)}
}

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {

    std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor

    WAYPOINT("UPTW");
    _llk_unpack_tilize_(
        base_address,
        tile_index,
        unpack_src_format[operand_id],
        block_ct_dim,
        face_r_dim,
        num_faces,
        narrow_tile
    );
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
inline void llk_unpack_tilizeA_B_hw_configure(const llk_unpack_AB_params_t *unpack_tilizeA_B_params, const int within_face_16x16_transpose = 0) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpack_tilizeA_B_params->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_tilizeA_B_params->unpB_operand);

    // unpA -> srcA
    // unpB -> srcB
    const uint32_t num_faces = get_operand_num_faces(unpA_operand_id);  // num faces in unpA and unpB are the same

    const uint32_t face_r_dim = get_operand_face_r_dim(unpA_operand_id);  // face r dim in unpA and unpB are the same

    _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        face_r_dim,
        within_face_16x16_transpose,
        num_faces);

}

template <bool is_fp32_dest_acc_en = false, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_tilizeA_B_hw_configure_disaggregated(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_AB_params_t unpack_tilizeA_B_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_tilizeA_B_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_tilizeA_B_params, within_face_16x16_transpose);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_mop_config(const bool narrow_tile = false, const std::uint32_t num_faces = 4) {
    static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, (zero_srcA ? 0b010001 : 0b1), 0, 0, 0, 1, (zero_srcA ? 0 : 1), p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb = TT_OP_UNPACR(SrcB, (zero_srcA ? 0b010001 : (reload_srcB ? 0b0 : 0b1)), 0, 0, 0, 1, (zero_srcA ? 0 : 1), p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Skip face ptr inc if same face is reloaded into srcB
    static constexpr uint unpack_neginf_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_NEGINFSRC); // Needed for max pool
    static constexpr uint unpack_zero_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);     // Needed for dot product
    static constexpr uint unpack_srcb_2_face = TT_OP_UNPACR(SrcB, 0b100010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product
    static constexpr uint unpack_srca_dat_valid = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product
    static constexpr uint unpack_srcb_dat_valid = TT_OP_UNPACR(SrcB, (reload_srcB ? 0b0 : 0b1), 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product

    const uint32_t innerloop = zero_srcA ? (num_faces>2 ? 2 : (num_faces-1)) : 1;
    const uint32_t outerloop = zero_srcA ? 1 : (num_faces>2) ? num_faces/2 : num_faces;
    ckernel_template tmp(outerloop, innerloop, unpack_srca, ((zero_srcA && num_faces==2) ? unpack_srcb_2_face : unpack_srcb));
    if constexpr (neginf_srcA) {
        tmp.set_start_op(unpack_neginf_srca);
    } else if constexpr (zero_srcA_reduce) {
        tmp.set_start_op(unpack_zero_srca);
    } else if constexpr (zero_srcA) {
        if (num_faces < 4) {
            tmp.set_start_op(unpack_zero_srca);
            tmp.set_end_ops(unpack_srca_dat_valid, unpack_srcb_dat_valid);
        }
    }
    tmp.program(instrn_buffer);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim, const std::uint32_t num_faces = 4, const std::uint32_t unpA_face_r_dim = FACE_R_DIM, const std::uint32_t unpB_face_r_dim = FACE_R_DIM) {

    const std::uint32_t operand_id = get_operand_id(operandA); // Use operandA to get operand_id tile dims must be the same for both operands
    //const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * ((narrow_tile || (num_faces == 1)) ? FACE_C_DIM : TILE_C_DIM);

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, unpA_face_r_dim*FACE_C_DIM-1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_face_r_dim*FACE_C_DIM-1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config = {0};
    config.f.out_data_format = unpack_dst_format[operand_id];
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE(unpack_src_format[operand_id], block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

    llk_unpack_tilizeA_B_mop_config<neginf_srcA,reload_srcB,zero_srcA,zero_srcA_reduce>(narrow_tile, num_faces);
}

template <bool zero_srcA = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4) {

    std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    //const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    std::uint32_t base_address_a = cb_interface[operandA_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format[operandA_id], tile_index_a) << (narrow_tile ? 0 : 1);
                                                    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
                                                    // For narrow tile we unpack 1 face in each iteration
                                                    // Offset address is in 16B words
                                                    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B = block_ct_dim * ((narrow_tile || (num_faces==1)) ? FACE_C_DIM/16 : TILE_C_DIM/16);
    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE(unpack_src_format[operandA_id], face_r_dim*block_c_dim_16B);  //*N rows / 16 to get 16B word aligned address


    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_b = cb_interface[operandB_id].fifo_rd_ptr - 1; // Remove header size added by descriptor
    std::uint32_t offset_address_b = tile_index_b*cb_interface[operandB_id].fifo_page_size;
    std::uint32_t address_b = base_address_b + offset_address_b;

    // Program srcA and srcB base addresses
    std::uint32_t num_loops = narrow_tile ? 2 : ((num_faces>1) ? num_faces/2 : 1);

    // Clear z/w start counters for SrcB
    TTI_SETADCZW(UNP1, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    WAYPOINT("UPTW");
    for (std::uint32_t n = 0; n < num_loops; n++) {
        std::uint32_t address_a = base_address_a + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        if constexpr (zero_srcA) {
            if (num_faces==4 && n==1) {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1011);
            } else {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
            }
        } else {
            TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
        }

        // Wait for free context
        wait_for_next_context(2);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Get tile address
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
        }

        // Run MOP
        if constexpr (zero_srcA) {
            if (num_faces==4) {
                if (n==0) {
                    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
                    ckernel::ckernel_template::run(instrn_buffer);
                } else {
                    ckernel::ckernel_template::run(instrn_buffer);
                    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);
                    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
                }
            } else {
                ckernel::ckernel_template::run(instrn_buffer);
            }
        } else {
            ckernel::ckernel_template::run(instrn_buffer);
        }

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
    std::uint32_t unpA_face_r_dim = FACE_R_DIM /*unused*/) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<zero_srcA>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces);
    }
}
