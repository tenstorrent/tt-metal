// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_mop_config_(const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    static constexpr uint unpack_srca     = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_zerosrca = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint unpack_srcb     = TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    const uint32_t outerloop     = num_faces;
    constexpr uint32_t innerloop = 1;
    ckernel_template tmp(outerloop, innerloop, unpack_zerosrca, unpack_srca);
    tmp.set_start_op(unpack_srcb);
    tmp.program();
}

template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_init_(const std::uint32_t within_face_16x16_transpose = 0, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(ReduceDim::REDUCE_ROW == dim ? !within_face_16x16_transpose : within_face_16x16_transpose);

    TTI_SETADCXX(p_setadc::UNP0, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP1, FACE_C_DIM - 1, 0x0);

    _llk_unpack_reduce_mop_config_<type, dim>(num_faces);
}

template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_(const std::uint32_t address)
{
    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure address
    _llk_unpack_configure_single_address_(address, cfg);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
