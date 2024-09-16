// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK TILIZE SRC A
*************************************************************************/

template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);

    _llk_unpack_tilize_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id]
    );
}

template <bool is_fp32_dest_acc_en = false /* unused */>
inline void llk_unpack_tilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand
    };
    llk_unpack_tilize_hw_configure(&unpack_tilize_params);
}

inline void llk_unpack_tilize_mop_config() {
    _llk_unpack_tilize_mop_config_();
}

inline void llk_unpack_tilize_init(const std::uint32_t operand, const std::uint32_t ct_dim) {

    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t src_format = (std::uint32_t)unpack_src_format[operand_id];
    std::uint32_t dst_format = (std::uint32_t)unpack_dst_format[operand_id];

    _llk_unpack_tilize_init_(src_format, dst_format, ct_dim);
}

inline void llk_unpack_tilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    std::uint32_t input = get_operand_id(operand);
    unpack_config_u config = {0};

    TTI_SETADCXX(p_setadc::UNP0, face_r_dim*FACE_C_DIM-1, 0x0);
    TTI_SETADCXX(p_setadc::UNP1, face_r_dim*FACE_C_DIM-1, 0x0);

    config.f.out_data_format = (uint)unpack_dst_format[input];
    config.f.throttle_mode = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16); //GPR preloaded with  16 | (16 << 16)}
}


inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t src_format = (uint)unpack_src_format[operand_id];

    WAYPOINT("UPTW");
    _llk_unpack_tilize_(
        base_address,
        tile_index,
        src_format,
        block_ct_dim
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

template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_unpack_tilizeA_B_hw_configure(const llk_unpack_AB_params_t *llk_unpack_tilizeA_B) {

    const uint32_t unpA_operand_id = get_operand_id(llk_unpack_tilizeA_B->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(llk_unpack_tilizeA_B->unpB_operand);
    configure_unpack_AB(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id]
    );
}

template <bool is_fp32_dest_acc_en = false /* unused */>
inline void llk_unpack_tilizeA_B_hw_configure_disaggregated(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const llk_unpack_AB_params_t llk_unpack_tilizeA_B = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_tilizeA_B_hw_configure(&llk_unpack_tilizeA_B);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool reuse_srcB = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_mop_config(const std::uint32_t num_faces) {
    static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb = TT_OP_UNPACR(SrcB, (reuse_srcB ? 0b010010 : (reload_srcB ? 0b0 : 0b1)), 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Skip face ptr inc if same face is reloaded into srcB
    static constexpr uint unpack_neginf_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr::UNP_NEGINFSRC); // Needed for max pool
    static constexpr uint unpack_zero_srcb = TT_OP_UNPACR_NOP(SrcB, p_unpacr::UNP_ZEROSRC); // Needed for dot product
    static constexpr uint unpack_srcb_no_dat_valid = TT_OP_UNPACR(SrcB, 0b010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // needed for dot product
    static constexpr uint unpack_zerosrca = TT_OP_UNPACR_NOP(SrcA, p_unpacr::UNP_ZEROSRC);  // needed for SUM reduction

    constexpr uint32_t innerloop = 1;
    const uint32_t outerloop = reuse_srcB ? 1 : ((num_faces>2) ? num_faces/2 : num_faces);
    if constexpr (reuse_srcB) {
        if (num_faces == 1) {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, unpack_srca);
            tmp.set_start_op(unpack_zero_srcb);
            tmp.program(instrn_buffer);
        } else {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb_no_dat_valid, unpack_srcb);
            tmp.set_start_op(unpack_zero_srcb);
            tmp.set_end_ops(unpack_srca, unpack_srca);
            tmp.program(instrn_buffer);
        }
    } else if constexpr (neginf_srcA) {
        ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srcb);
        tmp.set_start_op(unpack_neginf_srca);
        tmp.program(instrn_buffer);
    } else if constexpr (zero_srcA_reduce) {   // SUM reduction needs to zero out the srcA
        ckernel_template tmp(outerloop, innerloop, unpack_zerosrca, unpack_srcb);
        tmp.set_start_op(unpack_zerosrca);
        tmp.program(instrn_buffer);
    }
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool reuse_srcB = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim, const std::uint32_t num_faces = 4, const std::uint32_t unpA_face_r_dim = FACE_R_DIM, const std::uint32_t unpB_face_r_dim = FACE_R_DIM) {

    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t src_format_A = (std::uint32_t)unpack_src_format[operandA_id];
    std::uint32_t dst_format_A = (std::uint32_t)unpack_dst_format[operandA_id];

    const std::uint32_t block_c_dim = ct_dim * ((num_faces > 2) ? num_faces/2 : num_faces) * FACE_C_DIM;

    // Override default settings
    unpack_config_u config = {0};
    config.f.out_data_format = dst_format_A;
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE(src_format_A, block_c_dim)) >> 4;

    TTI_SETADCXX(p_setadc::UNP0, unpA_face_r_dim*FACE_C_DIM-1, 0x0);
    TTI_SETADCXX(p_setadc::UNP1, unpB_face_r_dim*FACE_C_DIM-1, 0x0);

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

    llk_unpack_tilizeA_B_mop_config<neginf_srcA,reload_srcB,reuse_srcB,zero_srcA_reduce>(num_faces);
}

template <bool reuse_srcB = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4) {

    //Setup SrcA unpack
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t base_address_a = cb_interface[operandA_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t unpack_srca_format = (uint)unpack_src_format[operandA_id];

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_srca_format, tile_index_a) << 1;
                                                   // Each iteration unpacks 2 16x16 faces (1st 0,1 2nd 2,3)
                                                   // Offset address is in 16B words
                                                   // Datum count = tile_index*16 (/16 to get word count)
    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE(unpack_srca_format, block_ct_dim*((num_faces > 2) ? num_faces/2 : num_faces));  //*16 rows / 16 to get 16B word aligned address

    //Set Tile address for Src B
    //Src B just does rowmajor unpack, with z counters incremented for every face
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_b = cb_interface[operandB_id].fifo_rd_ptr - 1; // Remove header size added by descriptor
    std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX<true>(unpack_src_format[operandB_id], tile_index_b);
    std::uint32_t address_b = base_address_b + offset_address_b;

    // Clear z/w start counters for SrcB
    TTI_SETADCZW(UNP1, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    WAYPOINT("UPTW");
    const std::uint32_t num_loops = (num_faces>1) ? num_faces/2 : 1;
    for (std::uint32_t n = 0; n < num_loops; n++) {
        std::uint32_t address_a = base_address_a + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);

        // Wait for free context
        wait_for_next_context(2);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Get tile address for src a
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
        }

        // Run MOP
        ckernel::ckernel_template::run(instrn_buffer);
        if (reuse_srcB && num_faces==4) {
            TTI_SETADCZW(UNP1, 0, 0, 0, 1, 0b0001);
        }

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
    WAYPOINT("UPTD");
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool reuse_srcB = false>
inline void llk_unpack_tilizeA_B_block(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t block_c_tiles_a,
    std::uint32_t tile_idx_b,
    std::uint32_t num_faces = 4,
    std::uint32_t unpA_face_r_dim = FACE_R_DIM /*unused*/) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<reuse_srcB>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces);
    }
}
