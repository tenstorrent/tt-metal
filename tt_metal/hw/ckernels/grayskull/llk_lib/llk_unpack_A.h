// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

#include "debug/dprint.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false>
inline void _llk_unpack_A_mop_config_(const bool transpose_of_faces) {

    if constexpr (BType == BroadcastType::COL) {
#if SKIP_UNP == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srca =
            TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif

        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        if constexpr (acc_to_dest) {
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
        static constexpr uint unpack_srca =
            TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
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
    }
    else if constexpr (BType == BroadcastType::SCALAR) {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
#if SKIP_UNP == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        ckernel_unpack_template tmp = ckernel_unpack_template::lB(unpack_srcb, TT_OP_NOP);
        tmp.program(instrn_buffer);
    }
    else {
        if (transpose_of_faces) {
            static constexpr uint unpack_srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001);
#if SKIP_UNP == 1
            static constexpr uint unpack_srca = TT_OP_NOP;
#else
            static constexpr uint unpack_srca =
                TT_OP_UNPACR(SrcA, 0b01000010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
            ckernel_unpack_template tmp = ckernel_unpack_template(
                false,  // src B
                true,   // halo - just used for 4 unpacks
                unpack_srca,
                unpack_srca,
                unpack_srca_set_z,
                TT_OP_NOP,
                0,
                0,
                0);
            tmp.program(instrn_buffer);
        } else {
            if constexpr (acc_to_dest) {
#if SKIP_UNP == 1
                static constexpr uint unpack_srca = TT_OP_NOP;
                static constexpr uint unpack_srcb = TT_OP_NOP;
#else
                static constexpr uint unpack_srca =
                    TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                static constexpr uint unpack_srcb =
                    TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

#endif
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
            } else {
#if SKIP_UNP == 1
               static constexpr uint unpack_srca = TT_OP_NOP;
#else
               static constexpr uint unpack_srca =
                   TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
               ckernel_unpack_template tmp = ckernel_unpack_template::lA(unpack_srca);
               tmp.program(instrn_buffer);
            }
        }
    }
}

inline void _llk_unpack_A_hw_configure_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const int transpose_xy = 0) {
    configure_unpack_AB(
        unpack_src_format,
        unpack_src_format,
        unpack_dst_format,
        unpack_dst_format
    );
}

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
// within_face_16x16_transpose is used on WH but not used for GS, this transpose is done in math on GS
inline void _llk_unpack_A_init_(const std::uint32_t transpose_of_faces=0, const std::uint32_t within_face_16x16_transpose=0) {
    _llk_unpack_A_mop_config_<BType, acc_to_dest>(transpose_of_faces);
}

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_unpack_A_(const std::uint32_t address, const int transpose_of_faces = 0) {

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
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest))  {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        } else {
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address;
        }
    } else {
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        } else {
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address;
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    if constexpr (acc_to_dest) {
        TTI_SETADCXX(p_setadc::UNP0, 0, 0x0);
    }

    if constexpr ((BType == BroadcastType::ROW) || ((BType == BroadcastType::COL) && acc_to_dest)) {
        mop_run(0, 2);
    } else if constexpr ((BType == BroadcastType::SCALAR) || (BType == BroadcastType::COL)) {
        mop_run(0, 1);
    } else if(transpose_of_faces) {
        mop_run(0, 2);
    } else {
        mop_run(0, 4);
    }

    if constexpr (acc_to_dest) {
        TTI_SETADCXX(p_setadc::UNP0, FACE_HEIGHT*16-1, 0x0);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

    #ifdef PERF_DUMP
        first_unpack_recorded = true;
    #endif

    DPRINT << "UNPA" << ENDL();
}
