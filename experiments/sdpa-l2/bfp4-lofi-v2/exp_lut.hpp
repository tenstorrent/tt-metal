// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef SDPA_LOFI_LUT_EXP
#include "exp_native.hpp"
#else
#ifndef SDPA_LOFI_NATIVE_EXP
#error "LUT refinement requires the native FP32 exp grid"
#endif

// Include this before compute_streaming.hpp. Its later exp_native.hpp include
// is then a pragma-once no-op; only these private wrappers change the call.
#define exp_native_packthread_tile exp_grid_only_packthread_tile
#include "exp_native.hpp"
#undef exp_native_packthread_tile

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {

template <int iterations>
inline void calculate_lofi_exp_lut2() {
    static_assert(iterations > 0 && iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);

    // Relative-LS approximation of exp2(m-1)/m, m in[1,2), split at1.5.
    // SFPLUTFP32 TABLE1 selects packed FP16 slope/intercept from L1/L5 in
    // precisely those intervals. Other table entries cannot be selected:
    // SETEXP below forces abs(m) into[1,2), including zero/underflow grid.
    // Source of constants: exp_lut_models.py, exp-lut-model-v1.jsonl.
    TTI_SFPLOADI(1, 0xA, 0xaee8);
    TTI_SFPLOADI(1, 0x8, 0x2f59);
    TTI_SFPLOADI(5, 0xA, 0x3c5f);
    TTI_SFPLOADI(5, 0x8, 0x3a1c);

    // Preserve the native exp replay0..7 and programmable constants12..14.
    // L0/L2 retain signed grid values; L3 is the LUT input; L4/L6 results.
    // The LUT takes abs(L3). Multiplication restores the grid sign, so
    // negative underflow values are still zeroed by the caller's pack ReLU.
    TTI_REPLAY(8, 10, 1, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(2, 0, ADDR_MOD_6, 2);
    TTI_SFPSETEXP(127, 0, 3, 1);
    TTI_SFPLUTFP32(4, 2);
    TTI_SFPSETEXP(127, 2, 3, 1);
    TTI_SFPLUTFP32(6, 2);
    TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
    TTI_SFPMUL(2, 6, p_sfpu::LCONST_0, 2, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
    TTI_SFPSTORE(2, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 10);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace ckernel {
template <int iterations>
ALWI void exp_native_packthread_tile(uint32_t idst) {
    exp_grid_only_packthread_tile<iterations>(idst);
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_lofi_exp_lut2, (iterations), idst, VectorMode::None)));
}
}  // namespace ckernel
#endif
