// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h"

namespace ckernel
{
namespace sfpu
{

// Seed a non-identity, bit-15-flagged caller mapping in face 0: id ^ 0x8055.
// Each DEST offset addresses the even or odd columns of a four-row band.
template <int offset = 0>
inline void _moe_gate_test_seed_indices_()
{
    static_assert(offset >= 0 && offset % 2 == 0, "offset must address the even or odd columns of a four-row band");
    if constexpr (offset < 16)
    {
        constexpr int first_id = (offset / 4) * 64 + (offset % 4) / 2;
        TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);
        TTI_SFPIADD(first_id, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x8055);
        TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, generic_moe_gate_indices_tile + offset);
        _moe_gate_test_seed_indices_<offset + 2>();
    }
    else
    {
        // One past face 0 is the valid recursion terminator.
        static_assert(offset == 16, "offset must not extend past face 0");
    }
}

} // namespace sfpu
} // namespace ckernel
