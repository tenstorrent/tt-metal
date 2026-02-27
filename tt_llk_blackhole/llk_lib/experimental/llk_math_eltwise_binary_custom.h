// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, std::uint32_t FIDELITY_INCREMENT>
inline void eltwise_binary_configure_addrmod_custom()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity (LoFi, HiFi2, HiFi3, HiFi4) for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type
 * @param num_faces: Number of faces to process (1, 2, or 4)
 * @param acc_to_dest: Accumulate to destination flag
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_custom_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    constexpr std::uint32_t math_fidelity_increment = 1;

    eltwise_binary_configure_addrmod_custom<eltwise_binary_type, src_b_bcast_type, math_fidelity_increment>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

inline void _llk_math_eltwise_binary_bcast_reuse_custom_(const std::uint32_t ct_dim = 1)
{
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 24},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0x3F & -8}, // decrement srcB by 8
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_5);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB); // reset both src counters to 0

    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        // F0 - F0
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // 0 -> 8
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_5, 0); // 8 -> 0

        // F1 - F0
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // 0 -> 8
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_6, 0); // 8 -> 32

        // F2 - F2
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // 32 -> 40
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_5, 0); // 40 -> 32

        // F3 - F2
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // 32 -> 40
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_6, 0); // 40 -> 64, no CLR_A

        // Reset srcB to 0 for next tile, but keep srcA advancing
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB); // reset only srcB counter to 0
    }

    // Final cleanup: reset both counters
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB);
}
