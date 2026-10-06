// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
}
#include "sfpu/ckernel_sfpu_bf16_reciprocal_complement_core.h"
#include "sfpu/ckernel_sfpu_bf16_reciprocal_complement.h"
namespace ckernel::sfpu::bf16 {
// Replay slots of one BH row: 12 ahead of the Horner rungs, two per rung (a gap slot,
// then the MAD), one more where degree seven reloads LREG3, and the 4-slot tail.
constexpr unsigned reciprocal_complement_body_slots(unsigned degree) {
    return degree == 8 ? 32u : 12u + 2u * (degree - 1u) + (degree == 7 ? 1u : 0u) + 4u;
}
template <class Config, int Iterations = 32>
inline void calculate_reciprocal_complement() {
    static_assert(Iterations == 32);
    // Degree four leaves LREG5 out of the replayed body, so it can carry the NaN rows.
    static_assert(Config::kDegree == 4);
    static_assert(Config::kBodySlots == reciprocal_complement_body_slots(Config::kDegree));
    // The body overwrites each row with its result and DEST past this tile holds the
    // caller's other tiles, so whether row d is NaN is kept as bit 31 - d of LREG5.
    sfpi::vInt nan_rows = 0;
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        sfpi::vFloat raw = sfpi::dst_reg[d];
        nan_rows <<= 1;
        v_if(sfpi::as<sfpi::vInt>(sfpi::setsgn(raw, 0)) > 0x7f800000) { nan_rows |= 1; }
        v_endif;
    }
    sfpi::l_reg[sfpi::LRegs::LReg5] = nan_rows;
    // The rungs read c2 and c1 from LREG7 and LREG6.
    ::ckernel::sfpu::bf16_sfpi::sfploadi(7, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[2] >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(7, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[2] & 0xffffu);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(6, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(6, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(4, sfpi::SFPLOADI_MOD0_UPPER, Config::kComplementBits >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(4, sfpi::SFPLOADI_MOD0_LOWER, Config::kComplementBits & 0xffffu);
    ::ckernel::sfpu::bf16_sfpi::replay(0, Config::kBodySlots, 1, 1);
    sfpi::reciprocal_complement_body<Config, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int d = 1; d < 32; ++d) {
        ::ckernel::sfpu::bf16_sfpi::replay(0, Config::kBodySlots, 0, 0);
    }
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    // The body's macro stored every row rounded to BF16, so only a NaN row is rewritten,
    // to +Inf for either sign. The traversal restarts at row 0, independently of caller cleanup.
    ::ckernel::math::clear_dst_reg_addr();
    sfpi::vInt nan_flags = sfpi::l_reg[sfpi::LRegs::LReg5];
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        v_if(nan_flags < 0) { sfpi::dst_reg[d] = sfpi::target_raw_terminal_value<1>(sfpi::vFloat(0.0f)); }
        v_endif;
        nan_flags <<= 1;
    }
}
}  // namespace ckernel::sfpu::bf16
