// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_() {
#if SKIP_UNP == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
    static constexpr uint unpack_srcb = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
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

inline void _llk_unpack_AB_hw_configure_(
    const std::uint32_t unpA_src_format, const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format, const std::uint32_t unpB_dst_format) {
    configure_unpack_AB(
        unpA_src_format,
        unpB_src_format,
        unpA_dst_format,
        unpB_dst_format
    );
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const std::uint32_t transpose=0 /* unused */, const std::uint32_t acc_to_dest=0 /* unused */) {
    _llk_unpack_AB_mop_config_<BType>();
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a, const std::uint32_t address_b, const bool transpose_of_faces = 0 /*not used*/) {

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

    // Stall unpacker until pending CFG writes from Trisc have completed
    // TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    if constexpr ((BType == BroadcastType::ROW) || (BType == BroadcastType::COL)) {
        mop_run(0, 2);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        mop_run(0, 1);
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
