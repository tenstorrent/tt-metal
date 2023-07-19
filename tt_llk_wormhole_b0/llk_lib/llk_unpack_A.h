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

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_A_mop_config(const bool transpose_of_faces, const std::uint32_t num_faces) {

    static_assert(!((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");

    if constexpr (BType == BroadcastType::COL) {
#if SKIP_UNP == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        // TODO: add support for srcB dest reuse
        static constexpr uint unpack_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC_SET_DVALID);
        static constexpr uint unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        
#endif

        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        if constexpr (acc_to_dest) {
            ckernel_unpack_template tmp = ckernel_unpack_template(
                false,
                true,   // halo - just used for 4 unpacks
                unpack_srcb,
                unpack_srca,
                unpack_srca,
                unpack_srcb_set_z,
                0,
                0,
                0);
            tmp.program(instrn_buffer);
        } else {
            ckernel_unpack_template tmp = ckernel_unpack_template(
                false,  // src B
                true,   // halo - just used for 4 unpacks
                unpack_srcb,
                unpack_srcb_set_z,
                unpack_srcb,
                unpack_srca,
                0,
                0,
                0);
            tmp.program(instrn_buffer);
        }
    } else if constexpr (BType == BroadcastType::ROW) {
#if SKIP_UNP == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        // TODO: add support for srcB dest reuse
        static constexpr uint unpack_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC_SET_DVALID);
        static constexpr uint unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        static constexpr uint unpack_srcb_clear_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        if constexpr (acc_to_dest) {
            ckernel_unpack_template tmp = ckernel_unpack_template(
                true ,  // src B
                true,   // halo - just used for 4 unpacks
                unpack_srcb,
                unpack_srca,
                unpack_srcb,
                unpack_srca,
                0,
                unpack_srcb_clear_z,
                0);
            tmp.program(instrn_buffer);

        } else {
            ckernel_unpack_template tmp = ckernel_unpack_template(
                false,  // src B
                true,   // halo - just used for 4 unpacks
                unpack_srcb,
                unpack_srcb,
                unpack_srcb_clear_z,
                TT_OP_NOP,
                0,
                0,
                0);
            tmp.program(instrn_buffer);
        }
    } else if constexpr (BType == BroadcastType::SCALAR) {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
#if SKIP_UNP == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        ckernel_unpack_template tmp = ckernel_unpack_template::lB(unpack_srcb, TT_OP_NOP);
        tmp.program(instrn_buffer);
    } else {
        if (transpose_of_faces) {
            constexpr uint replay_buf_len = 3;
            #if SKIP_UNP == 1
                TTI_REPLAY(0, 1, 0, 1);
                TTI_NOP;
                static constexpr uint unpack_srca_set_z = TT_OP_NOP;
            #else
                static constexpr uint unpack_srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001); // set srcA ch0_z = 1 
                TTI_REPLAY(0, replay_buf_len, 0, 1);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
                if (num_faces>2) {
                    TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=2
                } else {
                    TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=1
                }
            #endif 
            ckernel_unpack_template tmp = ckernel_unpack_template(
                num_faces<4 ? false : true, // src B
                num_faces<2 ? false : true,  // halo - just used for 4 unpacks
                TT_OP_REPLAY(0, replay_buf_len, 0, 0),                           // Unpack face 0 to srcA
                num_faces<2 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 2 or 1 to srcA
                num_faces<4 ? TT_OP_NOP : unpack_srca_set_z,
                num_faces<4 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 1 to srcA
                0,
                num_faces<4 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 3 to srcB
                0);
            tmp.program(instrn_buffer);
        } else {
            if constexpr (acc_to_dest) {
                #if SKIP_UNP == 1
                static constexpr uint unpack_srca = TT_OP_NOP;
                static constexpr uint unpack_srcb = TT_OP_NOP;
                static constexpr uint unpack_srcb_set_dvalid = TT_OP_NOP;
                #else
                static constexpr uint unpack_srca = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? 
                    TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC_SET_DVALID) : 
                    TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

                static constexpr uint unpack_srcb = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ?
                    TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC) : 
                    TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID); //WA for tenstorrent/budabackend#1230
                #endif
               ckernel_unpack_template tmp = ckernel_unpack_template(
                   !(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB),   // src B
                   (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB),  // halo - just used for 4 unpacks
                   unpack_srca,
                   unpack_srcb,
                   (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? unpack_srcb_set_dvalid : TT_OP_NOP,
                   TT_OP_NOP,
                   0,
                   unpack_srcb,
                   0);
               tmp.program(instrn_buffer);
            } else {
                constexpr uint replay_buf_len = 3;

                #if SKIP_UNP == 1
                    TTI_REPLAY(0, 1, 0, 1);
                    TTI_NOP;
                #else
                    TTI_REPLAY(0, replay_buf_len, 0, 1);
                    
                    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
                    TTI_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                #endif    

                ckernel_unpack_template tmp = ckernel_unpack_template(
                    false,  // src B
                    num_faces<2 ? false : true,  // halo - just used for 4 unpacks
                    TT_OP_REPLAY(0, replay_buf_len, 0, 0),
                    num_faces<2 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 1
                    num_faces<4 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 2
                    num_faces<4 ? TT_OP_NOP : TT_OP_REPLAY(0, replay_buf_len, 0, 0), // Unpack face 3
                    0,
                    0,
                    0);
        
                tmp.program(instrn_buffer);
            }
        }
    }
}

template <bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_A_hw_configure(const llk_unpack_A_params_t *unpack_A_params, const int within_face_16x16_transpose = 0) {
    constexpr bool is_row_pool = false;
    const uint32_t unpA_operand_id = get_operand_id(unpack_A_params->unpA_operand);

    const uint32_t unpA_num_faces = get_num_faces(unpA_operand_id);

    constexpr uint32_t unpA_face_height = 16;

    configure_unpack_AB(unpA_operand_id, unpA_operand_id,
        unpA_face_height, unpA_face_height, is_row_pool, within_face_16x16_transpose, is_fp32_dest_acc_en, srnd_fpu_en, unpA_num_faces, unpA_num_faces);
}

template <bool is_fp32_dest_acc_en = false, bool srnd_fpu_en = false>
inline void llk_unpack_A_hw_configure_disaggregated(const std::uint32_t unpA_operand, const int within_face_16x16_transpose = 0) {

    const llk_unpack_A_params_t unpack_A_params = {
        .unpA_operand = unpA_operand
    };
    llk_unpack_A_hw_configure<is_fp32_dest_acc_en, srnd_fpu_en>(&unpack_A_params, within_face_16x16_transpose);
}

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_A_init(const std::uint32_t transpose_of_faces=0, const std::uint32_t within_face_16x16_transpose=0, const std::uint32_t in_tile_dims[2] = default_tile_dims) {
    const std::uint32_t num_faces = get_num_faces(in_tile_dims);
    llk_unpack_A_mop_config<BType, acc_to_dest, binary_reuse_dest>(transpose_of_faces>0, num_faces);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);
    TTI_SETADCXX(0b11, FACE_WIDTH*FACE_HEIGHT-1, 0x0);
}

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index, const bool transpose_of_faces = 0 /*not used*/) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr;
    std::uint32_t offset_address = operands[input].f.tile_size_words * tile_index;
    std::uint32_t address = base_address + offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    if (0 == unp_cfg_context) {
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        } else {
            if constexpr(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) {
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
            }
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address;
        }
    } else {
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        } else {
            if constexpr(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) {
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
            }
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address;
        }
    }

    // Get num faces
    const std::uint32_t num_faces = get_num_faces(input);

    // Run MOP
    if constexpr ((BType == BroadcastType::ROW) || ((BType == BroadcastType::COL) && acc_to_dest)) {
        mop_run(0, (num_faces > 1) ? num_faces/2 : 1);
    } else if constexpr (acc_to_dest) {
        mop_run(0, num_faces);
    } else {
        mop_run(0, 1);
    } 

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
