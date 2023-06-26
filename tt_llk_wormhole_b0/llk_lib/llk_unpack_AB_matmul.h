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

// transpose is unused, math is adjusted to take into account srca face layout when transpose=true 
inline void llk_unpack_AB_matmul_mop_config(const bool transpose, const std::uint32_t ct_dim, const std::uint32_t rt_dim, const std::uint32_t kt_dim) {

    const bool reuse_a = ct_dim >= rt_dim;
    constexpr uint replay_buf_prog_len = 10;
    if (reuse_a) {
        #if SKIP_UNP == 1
            TTI_REPLAY(0, 1, 0, 1);
            TTI_NOP;
        #else
            TTI_REPLAY(0, replay_buf_prog_len, 0, 1);

            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG3_Base_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
            //TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG7_Offset_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP_LO);
            TTI_NOP;
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG3_Base_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
            //TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG7_Offset_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP_LO);
            TTI_NOP;
        #endif    
    } else {
        #if SKIP_UNP == 1
            TTI_REPLAY(0, 1, 0, 1);
            TTI_NOP;
        #else
            TTI_REPLAY(0, replay_buf_prog_len, 0, 1);

            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG3_Base_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
            //TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG7_Offset_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP_LO);
            TTI_NOP;
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG3_Base_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
            //TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG7_Offset_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP_LO);
            TTI_NOP;
        #endif    

    } 

    constexpr uint replay_buf_run_len = replay_buf_prog_len/2;
    static constexpr uint replay_context_0 =  TT_OP_REPLAY(                 0, replay_buf_run_len, 0, 0);
    static constexpr uint replay_context_1 =  TT_OP_REPLAY(replay_buf_run_len, replay_buf_run_len, 0, 0);
    
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,  // src B
        false,  // halo - just used for 4 unpacks
        replay_context_0, // runs when context is 0 
        0,
        0,
        0,
        replay_context_1, // run when context is 1
        0,
        0);
    
    tmp.program(instrn_buffer);
     

}

template<bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_AB_matmul_hw_configure(const llk_unpack_AB_matmul_params_t *unpack_AB_params) {
    constexpr uint32_t srca_height = 16;
    constexpr uint32_t srcb_height = 16;
    constexpr bool is_row_pool = false;
    const bool transpose_xy_srca = unpack_AB_params->transpose_xy_srca;
    configure_unpack_AB(get_operand_id(unpack_AB_params->unpB_operand), get_operand_id(unpack_AB_params->unpA_operand), 
                        srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en, srnd_fpu_en);

    // Override the default tile size to be 32x32
    TTI_SETADCXX(0b11, TILE_WIDTH*TILE_HEIGHT-1, 0x0);
}

template<bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_AB_matmul_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose_xy_srca = 0) {
    const llk_unpack_AB_matmul_params_t unpack_AB_matmul_params = {
        .unpA_operand = unpA_operand, .unpB_operand = unpB_operand, .transpose_xy_srca = transpose_xy_srca};
    llk_unpack_AB_matmul_hw_configure<is_fp32_dest_acc_en, srnd_fpu_en>(&unpack_AB_matmul_params);
}

inline void llk_unpack_AB_matmul_init(const std::uint32_t transpose=0, const std::uint32_t ct_dim=1, const std::uint32_t rt_dim=1, const std::uint32_t kt_dim=1) {
    llk_unpack_AB_matmul_mop_config(transpose != 0, ct_dim, rt_dim, kt_dim);
    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXX(0b11, TILE_WIDTH*TILE_HEIGHT-1, 0x0);

    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM)); // store kt_dim to gpr for scaling tile size
}

inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t tile_index_a, const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim=1, const std::uint32_t rt_dim=1, const std::uint32_t kt_dim=1) {

    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);
    std::uint32_t base_address_a = operands[inputA].f.fifo_rd_ptr;
    std::uint32_t base_address_b = operands[inputB].f.fifo_rd_ptr;
    const bool reuse_a = ct_dim >= rt_dim; 
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    if (!reuse_a) {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }

    for (uint t = 0; t < t_dim; t++) {

        std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], (tile_index_a + (reuse_a ? (t*kt_dim) : (0))));
        std::uint32_t next_offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], (tile_index_a + (reuse_a ? ((t+1)*kt_dim) : (0))));
        std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], (tile_index_b + (reuse_a ? (0       ) : (t))));
        std::uint32_t next_offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], (tile_index_b + (reuse_a ? (0       ) : (t+1))));
        std::uint32_t address_a = base_address_a + offset_address_a;
        std::uint32_t next_address_a = base_address_a + next_offset_address_a;
        std::uint32_t address_b = base_address_b + offset_address_b;
        std::uint32_t next_address_b = base_address_b + next_offset_address_b;

        // Wait for free context
        wait_for_next_context(2);

        semaphore_post(semaphore::UNPACK_SYNC);  // Trisc::SEMPOST for context acquire

        // Program unpacker 1 base address
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b; 
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_a;
        } 

        if (reuse_a) {
            #if SKIP_UNP == 1
                TTI_NOP;
            #else
                TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                if ((t+1)<t_dim) {
                    // Let's load one more tile into srcB
                    TT_SETDMAREG(0, LOWER_HALFWORD(next_address_a), 0, LO_16(p_gpr_unpack::TMP0));
                    TT_SETDMAREG(0, UPPER_HALFWORD(next_address_a), 0, HI_16(p_gpr_unpack::TMP0));
                    if (0 == unp_cfg_context) {
                        TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG3_Base_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                    } else {
                        TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG3_Base_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                    }    
                    TTI_DMANOP;
                    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    t++;
                }
            #endif    
        } else {
            #if SKIP_UNP == 1
                TTI_NOP;
            #else
                TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                if ((t+1)<t_dim) {
                    // Let's load one more tile into srcB
                    TT_SETDMAREG(0, LOWER_HALFWORD(next_address_b), 0, LO_16(p_gpr_unpack::TMP0));
                    TT_SETDMAREG(0, UPPER_HALFWORD(next_address_b), 0, HI_16(p_gpr_unpack::TMP0));
                    if (0 == unp_cfg_context) {
                        TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG3_Base_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                    } else {
                        TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG3_Base_cntx1_address_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                    }    
                    TTI_DMANOP;
                    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    t++;
                }
            #endif    
        }    

        TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xff); // Run the MOP

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }


    #ifdef PERF_DUMP
        first_unpack_recorded = true;
    #endif

}