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

inline void llk_unpack_AB_matmul_mop_config(bool transpose) {
    /*
    static constexpr uint unpack_srcb_top  = TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0,
    0, 0, 0, 1); static constexpr uint unpack_srcb_bot =  TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 1,
    p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1,
    1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); ckernel_unpack_template tmp  = ckernel_unpack_template(false, // src B
                                                            true, // halo - just used for 4 unpacks
                                                            unpack_srcb_top,
                                                            unpack_srcb_bot,
                                                            unpack_srca,
                                                            unpack_srca,
                                                            0, 0, 0);
    */

    //SrcB will be converted from row major to column major
    static constexpr uint unpack_srcb_set_dvalid =
        TT_OP_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);


    if(transpose){
        //SrcA unpacked as column major layout, follows src B
        static constexpr uint unpack_srca_set_dvalid =
            TT_OP_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        const uint replay_buf_len = 7;
        TTI_REPLAY(0, replay_buf_len, 0, 1);
        TTI_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_SETADCZW(0b011, 0, 0, 0, 1, 0b0001); // UNPACK SRC A and B Z 0,2,1,3
        TTI_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        ckernel_template tmp(1 /*outerloop*/, 1 /*innerloop*/, TT_OP_REPLAY(0, replay_buf_len, 0, 0));
        tmp.set_end_ops(unpack_srca_set_dvalid, unpack_srcb_set_dvalid);

        tmp.program(instrn_buffer);
    }else{
        //SrcA unpacked as row major layout
        static constexpr uint unpack_srca_set_dvalid =
            TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        const uint replay_buf_len = 7;
        TTI_REPLAY(0, replay_buf_len, 0, 1);
        TTI_UNPACR(SrcA,        0b1, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcA,        0b1, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_SETADCZW(0b010, 0, 0, 0, 1, 0b0001); // UNPACK SRCB Z 0,2,1,3
        TTI_UNPACR(SrcA,        0b1, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0 /*Don't Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        ckernel_template tmp(1 /*outerloop*/, 1 /*innerloop*/, TT_OP_REPLAY(0, replay_buf_len, 0, 0));
        tmp.set_end_ops(unpack_srca_set_dvalid, unpack_srcb_set_dvalid);

        tmp.program(instrn_buffer);
    }

}

template<bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_matmul_hw_configure(const llk_unpack_AB_matmul_params_t *unpack_AB_params) {
    constexpr uint32_t srca_height = 16;
    constexpr uint32_t srcb_height = 16;
    constexpr bool is_row_pool = false;
    bool transpose_xy_srca = unpack_AB_params->transpose_xy_srca;
    configure_unpack_AB(get_operand_id(unpack_AB_params->unpB_operand), get_operand_id(unpack_AB_params->unpA_operand),
                        srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en);
}

template<bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_matmul_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose_xy_srca = 0) {
    const llk_unpack_AB_matmul_params_t unpack_AB_matmul_params = {
        .unpA_operand = unpA_operand, .unpB_operand = unpB_operand, .transpose_xy_srca = transpose_xy_srca};
    llk_unpack_AB_matmul_hw_configure<is_fp32_dest_acc_en>(&unpack_AB_matmul_params);
}

inline void llk_unpack_AB_matmul_init(const std::uint32_t transpose=0) { llk_unpack_AB_matmul_mop_config(transpose != 0); }

inline void llk_unpack_AB_matmul(
    std::uint32_t operandA, std::uint32_t operandB, std::uint32_t tile_index_a, std::uint32_t tile_index_b) {
    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);
    std::uint32_t base_address_a = cb_interface[inputA].fifo_rd_ptr;
    std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], tile_index_a);
    std::uint32_t base_address_b = cb_interface[inputB].fifo_rd_ptr;
    std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], tile_index_b);

    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t address_a = base_address_a + offset_address_a - 1;
    std::uint32_t address_b = base_address_b + offset_address_b - 1;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Program srcA and srcB base addresses
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_a;
    }

    semaphore_post(semaphore::UNPACK_SYNC);  // Trisc::SEMPOST for context acquire

    // Stall unpacker until pending CFG writes from Trisc have completed
    // TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

#ifdef PERF_DUMP
    if (record_perf_events && !first_unpack_recorded) {
        uint32_t event_id_first_unpack = perf::get_event_id(
            0, 0, perf::EventType::UNPACK_FIRST_INSTRUCTION, current_outer_loop_iter);
        record_timestamp_64b(event_id_first_unpack);
        first_unpack_recorded = true;
    }
#endif

    // Run MOP
    ckernel_template::run(instrn_buffer);

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
