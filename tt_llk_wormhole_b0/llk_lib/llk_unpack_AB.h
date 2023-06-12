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
inline void llk_unpack_AB_mop_config(const bool transpose_of_faces=false) {
#if SKIP_UNP0 == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
#if SKIP_UNP1 == 1
    static constexpr uint unpack_srcb = TT_OP_NOP;
#else
    static constexpr uint unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif

    if constexpr (BType == BroadcastType::COL) {
        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            false,  // src B
            true,   // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srca,
            unpack_srca,
            unpack_srcb_set_z,
            0,
            0,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::ROW) {
        static constexpr uint unpack_srcb_clear_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,  // src B
            true,  // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srca,
            unpack_srcb,
            unpack_srca,
            0,
            unpack_srcb_clear_z,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,  // src B
            true,  // halo - just used for 4 unpacks
            unpack_srca,
            unpack_srca,
            unpack_srca,
            unpack_srca,
            0,
            unpack_srcb,
            0);
        tmp.program(instrn_buffer);
    } else {
        if (transpose_of_faces) {
            static constexpr uint srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001); // set z to 1
            static constexpr uint unpack_srca_skip_z =
                TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc z by 2
            ckernel_unpack_template tmp = ckernel_unpack_template(
                true,   // src B
                true,  // halo - just used for 4 unpacks
                unpack_srca_skip_z,
                unpack_srcb,
                unpack_srca_skip_z,
                unpack_srcb,
                0,
                srca_set_z,
                0);
            tmp.program(instrn_buffer);
        } else {
            ckernel_unpack_template tmp = ckernel_unpack_template(
                true,   // src B
                false,  // halo - just used for 4 unpacks
                unpack_srca,
                0,
                0,
                0,
                0,
                unpack_srcb,
                0);
            tmp.program(instrn_buffer);
        }    
    }
}

template <bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_AB_hw_configure(const llk_unpack_AB_params_t *unpack_AB_params, const int within_face_16x16_transpose = 0) {
    constexpr bool is_row_pool = false;
    constexpr uint32_t srca_height = 16;
    constexpr uint32_t srcb_height = 16;
    configure_unpack_AB(get_operand_id(unpack_AB_params->unpA_operand), get_operand_id(unpack_AB_params->unpB_operand), 
                            srca_height, srcb_height, is_row_pool, within_face_16x16_transpose, is_fp32_dest_acc_en, srnd_fpu_en);
}

template <bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_AB_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0 ) {
    const llk_unpack_AB_params_t unpack_AB_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_AB_hw_configure<is_fp32_dest_acc_en, srnd_fpu_en>(&unpack_AB_params, within_face_16x16_transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t transpose=0, const std::uint32_t acc_to_dest=0) {
    llk_unpack_AB_mop_config<BType>(transpose>0); // transpose of faces 0,2,1,3

    //Need to be able to configure tranpose srca for fused ops
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose); // transpose within the face

    TTI_SETADCXX(0b11, FACE_WIDTH*FACE_HEIGHT-1, 0x0);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t tile_index_a, const std::uint32_t tile_index_b, const bool transpose_of_faces = 0) {
    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);
    std::uint32_t base_address_a = operands[inputA].f.fifo_rd_ptr;
    std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], tile_index_a);
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = operands[inputB].f.fifo_rd_ptr;
    std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], tile_index_b);
    std::uint32_t address_b = base_address_b + offset_address_b;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

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
    if constexpr ((BType == BroadcastType::ROW) || (BType == BroadcastType::COL)) {
        mop_run(0, 2);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        mop_run(0, 1);
    } else if (transpose_of_faces) {
        mop_run(0, 2);
    } else {
        mop_run(0, 4);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
