// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef SDPA_LOFI_LUT_MACRO
#include "exp_lut.hpp"
#else
#if !defined(SDPA_LOFI_LUT_EXP) || !defined(SDPA_LOFI_NATIVE_EXP)
#error "Macro LUT refinement requires LUT and native-grid exp"
#endif

#define exp_native_packthread_tile exp_grid_only_packthread_tile
#include "exp_native.hpp"
#undef exp_native_packthread_tile

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {

template <int iterations>
inline void calculate_lofi_exp_lut2_macro() {
    static_assert(iterations > 0 && iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);

    // Same bits, table mode, SETEXP and final multiply as exp_lut.hpp.
    TTI_SFPLOADI(1, 0xA, 0xaee8);
    TTI_SFPLOADI(1, 0x8, 0x2f59);
    TTI_SFPLOADI(5, 0xA, 0x3c5f);
    TTI_SFPLOADI(5, 0x8, 0x3a1c);

    // Backdoor writes instruction templates, NOT programmable L12/13/14.
    // Native unclamped grid does not use template0 or Sequence1/2.
    TTI_SFPSETEXP(127, 0, 12, 1);               // template0: SETEXP via VC override
    TTI_SFPLUTFP32(13, 2);                      // template1: FP16 TABLE1 LUT
    TTI_SFPMUL(3, 0, p_sfpu::LCONST_0, 14, 0);  // template2: L3 * loaded VB

    // Sequence1: SETEXP delay0 (0x04), LUT delay1 (0x0d), result in loaded L3.
    // TABLE1 reads fixed L3; after SETEXP, only L1/L5 can be selected.
    TTI_SFPLOADI(0, 0xA, 0x0d04);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);
    // Sequence2: MUL delay0, loaded VB/VD (0x86); STORE delay2 (0x13).
    // Stores capture the load address and do not apply an address modifier.
    TTI_SFPLOADI(0, 0xA, 0x8600);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 6, 0);

    // Native init has Misc=0xf00: all delays count elapsed SFPU instructions.
    // Per vector issued at t=0..3: LM1, NOP, NOP, LM2.
    // Scheduled SETEXP t1, LUT t2, MUL t4, STORE t6. Next vector's
    // SETEXP t5, LUT t6 do not collide; L3 is read by MUL before next load.
    // A full SFPU instruction separates each MAD producer and consumer.
    // Only loads/NOPs are issued while macros run, avoiding silent subunit
    // collision drops. The final reload alone advances DST by four rows.
    // Replay0..7 and native Sequence0/template3 remain untouched.
    TTI_REPLAY(8, 8, 1, 1);
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_6, 0);  // Sequence1, L3, first vector
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_6, 0);  // Sequence2, L0
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_6, 2);  // Sequence1, L3, second vector
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_7, 2);  // Sequence2, L0, advance DST
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 8);
    }
    // Drain the final MUL/STORE before template updates or caller packing.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    restore_sdpa_grid_macro_instructions();
}

}  // namespace ckernel::sfpu
#endif

namespace ckernel {
template <int iterations>
ALWI void exp_native_packthread_tile(uint32_t idst) {
    exp_grid_only_packthread_tile<iterations>(idst);
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_lofi_exp_lut2_macro, (iterations), idst, VectorMode::None)));
}
}  // namespace ckernel
#endif
