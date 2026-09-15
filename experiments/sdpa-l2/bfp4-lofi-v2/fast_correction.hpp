// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include after the frozen FAST compute_common.hpp and before its streaming
// header. Keep the four selected source snapshots unchanged.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
template <uint32_t scale_fp32>
inline void calculate_sdpa_exp_correction_reset() {
    // Compensation's PACK SFPU replay leaves automatic destination increment
    // enabled. This SFPI routine advances dst_reg explicitly.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    calculate_sdpa_exp_correction<scale_fp32>();
}
}  // namespace ckernel::sfpu
#endif
#define calculate_sdpa_exp_correction calculate_sdpa_exp_correction_reset
