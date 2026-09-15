// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
inline void calculate_lofi_normalize_bias() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat out = sfpi::dst_reg[0];
        sfpi::vFloat inverse = sfpi::dst_reg[32];
        sfpi::vFloat bias = sfpi::dst_reg[64];
        sfpi::dst_reg[0] = out * inverse + bias;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
