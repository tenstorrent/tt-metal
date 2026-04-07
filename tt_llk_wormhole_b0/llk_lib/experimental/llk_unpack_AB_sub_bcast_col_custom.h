// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

// Custom init for the blocked sub+bcast(col) unpack flow.
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_sub_bcast_col_init_custom_(
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0); // transpose within the face

    // Force both unpackers to unpack the full 32x32 tile for the blocked path.
    TTI_SETADCXX(p_setadc::UNP0, 1023, 0x0);
    TTI_SETADCXX(p_setadc::UNP1, 1023, 0x0);
}

// Custom blocked unpack: one SrcB tile + ct_dim SrcA tiles.
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_sub_bcast_col_custom_(const std::uint32_t address_a, const std::uint32_t address_b, const std::uint32_t ct_dim = 1)
{
    // Start from a clean A/B tile index for each blocked sub call.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program SrcA and SrcB base addresses.
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
    // Wait for a free unpack config context before programming the next block.
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    // Trisc::SEMPOST for context acquire.
    semaphore_post(semaphore::UNPACK_SYNC);
    // Stall unpacker until pending CFG writes from Trisc have completed.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;

    // Load SrcB once and keep reusing it for all ct_dim SrcA tiles.
    TTI_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    for (std::uint32_t tile = 0; tile < ct_dim; tile++)
    {
        TTI_UNPACR(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 1, 0);
    }

    // T6::SEMGET for context release.
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    // Switch unpacker config context so the next blocked call can program the
    // other side while this one executes.
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_AB_sub_bcast_col_uninit_custom_()
{
    // Restore the default full-face unpack X span used by the generic unpack
    // helpers. The custom blocked path forces both unpackers to full 32x32.
    TTI_SETADCXX(p_setadc::UNP_AB, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}
