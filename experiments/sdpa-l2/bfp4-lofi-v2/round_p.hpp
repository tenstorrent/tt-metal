// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "sfpi.h"
namespace ckernel::sfpu {
template <int ITERATIONS>
inline void calculate_lofi_round_p() {
    // Exp's hand-written replay leaves an auto-incrementing modifier active;
    // SFPI advances dst_reg explicitly and requires zero automatic increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr int shift = 17;  // Seven significant bits, exactly consumed by SrcB.
#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vInt raw = sfpi::as<sfpi::vInt>(x);
        sfpi::vInt odd = (raw >> shift) & 1;
        raw = (raw + 65535 + odd) & -131072;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(raw);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
