#pragma once
#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_mop_config(const bool transpose_of_faces=false, const std::uint32_t operand_id=0) {
#if SKIP_UNP == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
    static constexpr uint unpack_srcb = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    const uint32_t num_faces = get_num_faces(operand_id); 
    const bool narrow_tile = get_narrow_tile(operand_id); // if narrow tile read face 0 twice for row broadcast
                                                          // or read face 0 and 1 for col broadcast

    if constexpr (BType == BroadcastType::COL) {
        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        const uint32_t outerloop = num_faces < 4 ? 1 : 2;   
        const uint32_t innerloop = num_faces < 2 ? 1 : 2;   
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb);
        if (narrow_tile) {
            tmp.set_end_op(unpack_srcb); // Read face 1
        } else {
            tmp.set_end_op(unpack_srcb_set_z);
        }    
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::ROW) {
        static constexpr uint unpack_srcb_clear_z  = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        static constexpr uint unpack_srcb_no_z_inc = TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        const uint32_t outerloop = num_faces < 4 ? 1 : 2;   
        const uint32_t innerloop = num_faces < 2 ? 1 : 2;   
        ckernel_template tmp(outerloop, innerloop, narrow_tile ? unpack_srcb_no_z_inc : unpack_srcb, unpack_srca);
        tmp.set_end_op(unpack_srcb_clear_z);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        const uint32_t outerloop = 1;   
        const uint32_t innerloop = num_faces;   
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb);
        tmp.program(instrn_buffer);
    } else {
        if (transpose_of_faces) {
            static constexpr uint srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001); // set z to 1
            static constexpr uint unpack_srca_skip_z =
                TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc z by 2
            const uint32_t outerloop = num_faces < 4 ? 1 : 2;   
            const uint32_t innerloop = num_faces < 2 ? 1 : 2;   
            ckernel_template tmp(outerloop, innerloop, num_faces<4 ? unpack_srca : unpack_srca_skip_z, unpack_srcb);
            tmp.set_end_op(srca_set_z);
            tmp.program(instrn_buffer);
        } else {
            constexpr uint32_t outerloop = 1;   
            const uint32_t innerloop = num_faces;   
            ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srcb);
            tmp.program(instrn_buffer);
        }    
    }

}

template <bool is_fp32_dest_acc_en = false, StochRndMode stoch_rnd_mode = StochRndMode::None>
inline void llk_unpack_AB_hw_configure(const llk_unpack_AB_params_t *unpack_AB_params, const int within_face_16x16_transpose = 0) {
    constexpr bool is_row_pool = false;
    // In0 -> unpA 
    // In1 -> unpB 
    const uint32_t unpA_operand_id = get_operand_id(unpack_AB_params->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_AB_params->unpB_operand);

    // unpA -> srcA
    // unpB -> srcB
    const uint32_t num_faces = get_num_faces(unpA_operand_id);  // num faces in unpA and unpB are the same
 
    const uint32_t face_r_dim = get_face_r_dim(unpA_operand_id); // face r dim in unpA and unpB are the same

    configure_unpack_AB<is_row_pool, is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpA_operand_id, 
        unpB_operand_id, 
        face_r_dim, 
        face_r_dim, 
        within_face_16x16_transpose, 
        num_faces, 
        num_faces);
}

template <bool is_fp32_dest_acc_en = false, StochRndMode stoch_rnd_mode = StochRndMode::None>
inline void llk_unpack_AB_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0 ) {

    TT_LLK_DUMP("llk_unpack_AB_hw_configure_disaggregated<{}, {}>({}, {}, {})", is_fp32_dest_acc_en, (uint8_t)stoch_rnd_mode, unpA_operand, unpB_operand, within_face_16x16_transpose);
    
    const llk_unpack_AB_params_t unpack_AB_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_AB_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_AB_params, within_face_16x16_transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose=0, const std::uint32_t acc_to_dest=0) {
    TT_LLK_DUMP("llk_unpack_AB_init<{}>({}, {}, {}, {})", BType, unpA_operand, unpB_operand, transpose, acc_to_dest);

    const uint32_t unpA_operand_id = get_operand_id(unpA_operand);

    //Need to be able to configure tranpose srca for fused ops
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose); // transpose within the face

    const uint32_t face_r_dim = get_face_r_dim(unpA_operand_id); // face r dim in unpA and unpB are the same

    constexpr std::uint32_t UNP_SEL = p_setadc::UNP_AB;
    config_face_dim<false, UNP_SEL>(face_r_dim);

    llk_unpack_AB_mop_config<BType>(transpose>0, unpA_operand_id); // transpose of faces 0,2,1,3
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t tile_index_a, const std::uint32_t tile_index_b, const bool transpose_of_faces = 0 /*not used*/) {
    TT_LLK_DUMP("llk_unpack_AB<{}>({}, {}, {}, {}, {}, {})", BType, operandA, operandB, tile_index_a, tile_index_b, transpose_of_faces);
    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);
    std::uint32_t base_address_a = operands[inputA].f.fifo_rd_ptr;
    std::uint32_t offset_address_a = operands[inputA].f.tile_size_words * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = operands[inputB].f.fifo_rd_ptr;
    std::uint32_t offset_address_b = operands[inputB].f.tile_size_words * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

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
