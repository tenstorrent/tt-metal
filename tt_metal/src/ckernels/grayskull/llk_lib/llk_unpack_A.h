/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

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

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
inline void llk_unpack_A_mop_config() {

    static_assert((!(acc_to_dest && transpose_xy)) && "accumulate into dest with transpose xy is not supported!");
    static_assert((!(transpose_xy && (BType != BroadcastType::NONE))) && "transpose xy with broadcast is not supported!");

    if constexpr (BType == BroadcastType::COL) {
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
#if SKIP_UNP1 == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        ckernel_unpack_template tmp = ckernel_unpack_template::lB(unpack_srcb, TT_OP_NOP);
        tmp.program(instrn_buffer);
    } else {
        if constexpr (transpose_xy) {
            static constexpr uint unpack_srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001);
#if SKIP_UNP0 == 1
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
#if SKIP_UNP0 == 1
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

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool transpose_xy_srca = false, bool acc_to_dest = false>
inline void llk_unpack_A_hw_configure(const llk_unpack_A_params_t *unpack_A_params) {

    static_assert((!(acc_to_dest && transpose_xy)) && "accumulate into dest with transpose xy is not supported!");
    static_assert((!(transpose_xy && (BType != BroadcastType::NONE))) && "transpose xy with broadcast is not supported!");

    configure_unpack_AB(
        get_operand_id(unpack_A_params->unpA_operand), get_operand_id(unpack_A_params->unpA_operand));
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool transpose_xy_srca = false, bool acc_to_dest = false>
inline void llk_unpack_A_hw_configure_disaggregated(const std::uint32_t unpA_operand) {

    const llk_unpack_A_params_t unpack_A_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_A_hw_configure<BType, transpose_xy, transpose_xy_srca, acc_to_dest>(&unpack_A_params);
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
// within_face_16x16_transpose is used on WH but not used for GS, this transpose is done in math on GS
inline void llk_unpack_A_init(const std::uint32_t within_face_16x16_transpose=0) {
    llk_unpack_A_mop_config<BType, transpose_xy, acc_to_dest>();
}



template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, bool transpose_of_faces = false>
inline void llk_unpack_A_mop_config_cm() {

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

// template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, int transpose_of_faces = 0>
ALWI void llk_unpack_A_cm(const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {

    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[input].fifo_rd_ptr - 1;

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {

        std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
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
        } else if constexpr (transpose_of_faces) {
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
    }
}
// // template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
// template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, int transpose_of_faces = 0>
// ALWI void llk_unpack_A_cm(const std::uint32_t operand, const std::uint32_t tile_index) {
//     // TT_LLK_DUMP("llk_unpack_A<{}, {}, {}>({}, {}, {})", BType, acc_to_dest, binary_reuse_dest, operand, tile_index, transpose_of_faces);
//     std::uint32_t input = get_operand_id(operand);
//     std::uint32_t base_address = cb_interface[input].fifo_rd_ptr - 1;
//     std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
//     std::uint32_t address = base_address + offset_address;

//     // Clear z/w start counters
//     TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

//     // Program srcA and srcB base addresses
//     volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

//     // Wait for free context
//     wait_for_next_context(2);

//     // Trisc::SEMPOST for context acquire
//     semaphore_post(semaphore::UNPACK_SYNC);

//     // Get tile address
//     if (0 == unp_cfg_context) {
//         if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest))  {
//             cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
//         } else {
//             cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address;
//         }
//     } else {
//         if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
//             cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
//         } else {
//             cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address;
//         }
//     }

//     // Stall unpacker until pending CFG writes from Trisc have completed
//     TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

//     // Run MOP
//     if constexpr (acc_to_dest) {
//         TTI_SETADCXX(p_setadc::UNP0, 0, 0x0);
//     }

//     if constexpr ((BType == BroadcastType::ROW) || ((BType == BroadcastType::COL) && acc_to_dest)) {
//         mop_run(0, 2);
//     } else if constexpr ((BType == BroadcastType::SCALAR) || (BType == BroadcastType::COL)) {
//         mop_run(0, 1);
//     } else if(transpose_of_faces) {
//         mop_run(0, 2);
//     } else {
//         mop_run(0, 4);
//     }

//     if constexpr (acc_to_dest) {
//         TTI_SETADCXX(p_setadc::UNP0, FACE_HEIGHT*16-1, 0x0);
//     }

//     // T6::SEMGET for context release
//     t6_semaphore_get(semaphore::UNPACK_SYNC);

//     // Switch unpacker config context
//     switch_config_context(unp_cfg_context);

// #ifdef PERF_DUMP
//     first_unpack_recorded = true;
// #endif
// }

template <BroadcastType BType = BroadcastType::NONE, bool acc_to_dest = false, std::uint32_t transpose_of_faces = 0>
ALWI void llk_unpack_A_init_cm(const std::uint32_t within_face_16x16_transpose=0, const std::uint32_t operand = 255) {
    // TT_LLK_DUMP("llk_unpack_A_init<{}, {}, {}>({}, {}, {})", BType, acc_to_dest, binary_reuse_dest, transpose_of_faces, within_face_16x16_transpose, operand);
    // Todo: figure out tile dims.
    // If passed in operand is default (255), it means that it has not been passed by llk, and we should assume default tile dims.
    llk_unpack_A_mop_config_cm<BType, acc_to_dest, transpose_of_faces>();
}


template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
inline void llk_unpack_A(std::uint32_t operand, std::uint32_t tile_index) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[input].fifo_rd_ptr;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t address = base_address + offset_address - 1;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

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

    if constexpr (acc_to_dest) {
        TTI_SETADCXX(p_setadc::UNP0, 0, 0x0);
    }

    // Run MOP
    if constexpr ((BType == BroadcastType::ROW) || (transpose_xy) || ((BType == BroadcastType::COL) && acc_to_dest)) {
        mop_run(0, 2);
    } else if constexpr ((BType == BroadcastType::SCALAR) || (BType == BroadcastType::COL)) {
        mop_run(0, 1);
    } else {
        mop_run(0, 4);
    }

    if constexpr (acc_to_dest) {
        // Don't need extra because we don't have a tile header?
        // TTI_SETADCXX(p_setadc::UNP0, FACE_HEIGHT*16-1, 0x0);
        TTI_SETADCXX(p_setadc::UNP0, FACE_HEIGHT*16, 0x0);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
