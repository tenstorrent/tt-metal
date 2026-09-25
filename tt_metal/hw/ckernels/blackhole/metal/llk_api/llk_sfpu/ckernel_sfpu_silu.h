// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// SILU_BF16_FAST_ALLOW_DST_SPILL: the fast kernel below spills one intermediate per vector to dst rows 64..78
// relative to the current face base, i.e. into the NEXT dst tile's region (and for face 3 of tile 7 in SyncHalf
// 16-bit dest, into the other half). That is harmless for the single-tile-in-dst usage it was validated in
// (ttnn eltwise unary at dst index 0), but it corrupts any caller that keeps live tiles at idst+1 (e.g. the
// moe_compute SWIGLU kernel runs silu on dst 0 and 2 while dst 1 and 3 hold the up-projection). Until a
// spill-free schedule is validated, the fast path is therefore opt-in.
#if !defined(DISABLE_SFPLOADMACRO) && defined(SILU_BF16_FAST_ALLOW_DST_SPILL)
// =====================================================================================================================
// Fast bf16 silu for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 859.1 cycles/tile vs 1298.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// State programmed by _init_silu_bf16_fast_: LREG11..14 (config path) and LREG6/7; SFPLOADMACRO instruction
// template 0 (SFP_STOCH_RND backdoor), macro sequence 0 (SFPCONFIG dest 4) and the LoadMacro Misc register
// (dest 8, StoreMod0 = FP16B). No replay slots. Needs ADDR_MOD_7 = {0,0,0} (re-asserted by the init). bf16 DEST
// only: never use for fp32 dest or on Wormhole. Processes one face (8 dst vectors at ADDR_MOD_7 offsets
// 0,2,...,14) per call and SPILLS to dst rows 64+offset (see SILU_BF16_FAST_ALLOW_DST_SPILL above).
//
// silu(x) = x * sigmoid(x) for bf16, Blackhole SFPU. Hand-scheduled TTI.
//
// Math (identical to the validated sfpi v13 kernel, max ULP = 1):
//   u  = x*2^-63 clamped >= -95*2^-63          (K1 swap-clamp vs LREG11)
//   t  = (-x/ln2 + 64)*2^14 + 2^23, clamped >= 2^23 (identity zone x > ~43.7)
//   zf = bits(t)<<9 as float; m = setexp(zf,127) in [1,2)
//   h  = E0 + m*(E1 + m*E2)  ~ 2^(m-1)/m       (deg-2, 0.35% max rel err)
//   w  = zf*h + 2^-63;  s = sigmoid(x)*2^63 via SFPARECIP + 1 Newton step
//   y  = u*s;  tiny |x| (u flushed) takes a predicated y = x*0.50244 path.
//
// Schedule: 8-vector software pipeline per face. Each loop body braids
// vector k's phase-2 (poly/recip/CC/final) with vector k+1's phase-1
// (load/clamp/magic-add/shift) so every 2-cycle latency gap is filled.
// uc spills to the tile-1 DST region (row offset 64+) to fit registers.
// SFPLOADMACRO on the predicated x-reload schedules STOCHRND (round column)
// and the store (store column, carrier address) in parallel with the issue
// stream.
//
// Registers: L0 = q (phase-2 scratch: arecip/Newton/y), even vectors r=L1
// m=L2, odd r=L3 m=L4, L5 = shared transient (h1 / KT clamp const).
// Constants: L11 = -95*2^-63, L12 = E0, L13 = -2^77/ln2, L14 = 2^-63,
// L6 = E1, L7 = E2 (user regs, read-only here), L9 = 0, L10 = 1 (hw).
// =====================================================================================================================
inline void _init_silu_bf16_fast_() {
    // Self-contained: re-assert the common SFPU state (config reg + ADDR_MOD_7 + RWC reset) as exp_init does,
    // then program this kernel's constants and SFPLOADMACRO template/sequence.
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // ---- constant registers (write L0, then SFPCONFIG dest = LREG index)
    TTI_SFPLOADI(0, 2, 0x0000);  // K1 = -95*2^-63 = 0xA33E0000
    TTI_SFPLOADI(0, 8, 0xA33E);
    TTI_SFPCONFIG(0, 11, 0);
    TTI_SFPLOADI(0, 2, 0x8130);  // E0 = 1.45706747 = 0x3FBA8130
    TTI_SFPLOADI(0, 8, 0x3FBA);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 2, 0xAA3B);  // P0 = -2^77/ln2 = 0xE638AA3B
    TTI_SFPLOADI(0, 8, 0xE638);
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 2, 0x0000);  // P2 = 2^-63 = 0x20000000
    TTI_SFPLOADI(0, 8, 0x2000);
    TTI_SFPCONFIG(0, 14, 0);
    // ---- persistent user-register constants
    TTI_SFPLOADI(6, 2, 0xB7DB);  // E1 = -0.69421164 = 0xBF31B7DB
    TTI_SFPLOADI(6, 8, 0xBF31);
    TTI_SFPLOADI(7, 2, 0x4F0E);  // E2 = 0.2337 = 0x3E6F4F0E
    TTI_SFPLOADI(7, 8, 0x3E6F);
    // ---- InstructionTemplate[0] = STOCHRND fp32->bf16 on L0 (VD=12 programs)
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 12, 1);
    // ---- Macro 0: Round = template0 delay 5; Store = builtin delay 7
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = (5u << 3) | 4u;
        constexpr std::uint32_t store_bits = (7u << 3) | 3u;
        TTI_SFPLOADI(0, 10, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, 8, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0);
    }
    // Misc: StoreMod0 = 2 (FP16B), instruction-counted delays on all units
    TTI_SFPCONFIG(0xF02, 8, 1);
    TTI_SFPNOP;
}

// Phase-1 of the first vector (fills the pipeline).
#define SILU_FAST_PROLOGUE()          \
    TTI_SFPLOAD(1, 0, 7, 0);          \
    TTI_SFPMAD(1, 14, 9, 1, 0);       \
    TTI_SFPNOP;                       \
    TTI_SFPSWAP(0, 1, 11, 1);         \
    TTI_SFPNOP;                       \
    TTI_SFPSTORE(1, 2, 7, 64);        \
    TTI_SFPMAD(1, 13, 9, 1, 0);       \
    TTI_SFPADDI(0x4B10, 1, 0);        \
    TTI_SFPLOADI(5, 0, 0x4B00);       \
    TTI_SFPSWAP(0, 1, 5, 1);          \
    TTI_SFPNOP;                       \
    TTI_SFPSHFT(9, 1, 1, 5);          \
    TTI_SFPNOP;                       \
    TTI_SFPSETEXP(0x7F, 1, 2, 1);

// Steady-state body: phase-2 of vector k (regs rk/mk, dst offset offk)
// braided with phase-1 of vector k+1 (regs rn/mn, offset offn).
#define SILU_FAST_BODY(rk, mk, rn, mn, offk, offn) \
    TTI_SFPMAD(7, mk, 6, 5, 0);                    \
    TTI_SFPLOAD(rn, 0, 7, offn);                   \
    TTI_SFPMAD(5, mk, 12, mk, 0);                  \
    TTI_SFPMAD(rn, 14, 9, rn, 0);                  \
    TTI_SFPMAD(rk, mk, 14, rk, 0);                 \
    TTI_SFPSWAP(0, rn, 11, 1);                     \
    TTI_SFPARECIP(0, rk, 0, 0);                    \
    TTI_SFPSTORE(rn, 2, 7, 64 + offn);             \
    TTI_SFPMAD(rn, 13, 9, rn, 0);                  \
    TTI_SFPMAD(rk, 0, 10, rk, 1);                  \
    TTI_SFPADDI(0x4B10, rn, 0);                    \
    TTI_SFPMAD(0, rk, 0, rk, 0);                   \
    TTI_SFPLOADI(5, 0, 0x4B00);                    \
    TTI_SFPLOAD(0, 0, 7, 64 + offk);               \
    TTI_SFPSWAP(0, rn, 5, 1);                      \
    TTI_SFPABS(0, 0, mk, 1);                       \
    TTI_SFPSHFT(9, rn, rn, 5);                     \
    TTI_SFPSETCC(0, mk, 0, 6);                     \
    TT_SFPLOADMACRO(0, 0, 7, offk);                \
    TTI_SFPLOADI(rk, 1, 0x3805);                   \
    TTI_SFPENCC(3, 0, 0, 10);                      \
    TTI_SFPMAD(0, rk, 9, 0, 0);                    \
    TTI_SFPSETEXP(0x7F, rn, mn, 1);

// Phase-2 of the last vector (drains the pipeline).
#define SILU_FAST_EPILOGUE()          \
    TTI_SFPMAD(7, 4, 6, 5, 0);        \
    TTI_SFPMAD(5, 4, 12, 4, 0);       \
    TTI_SFPMAD(3, 4, 14, 3, 0);       \
    TTI_SFPARECIP(0, 3, 0, 0);        \
    TTI_SFPMAD(3, 0, 10, 3, 1);       \
    TTI_SFPMAD(0, 3, 0, 3, 0);        \
    TTI_SFPLOAD(0, 0, 7, 64 + 14);    \
    TTI_SFPABS(0, 0, 4, 1);           \
    TTI_SFPSETCC(0, 4, 0, 6);         \
    TT_SFPLOADMACRO(0, 0, 7, 14);     \
    TTI_SFPLOADI(3, 1, 0x3805);       \
    TTI_SFPENCC(3, 0, 0, 10);         \
    TTI_SFPMAD(0, 3, 9, 0, 0);        \
    TTI_SFPNOP;                       \
    TTI_SFPNOP;                       \
    TTI_SFPNOP;                       \
    TTI_SFPNOP;

inline void _calculate_silu_bf16_fast_() {
    SILU_FAST_PROLOGUE();
    SILU_FAST_BODY(1, 2, 3, 4, 0, 2);    // k=0
    SILU_FAST_BODY(3, 4, 1, 2, 2, 4);    // k=1
    SILU_FAST_BODY(1, 2, 3, 4, 4, 6);    // k=2
    SILU_FAST_BODY(3, 4, 1, 2, 6, 8);    // k=3
    SILU_FAST_BODY(1, 2, 3, 4, 8, 10);   // k=4
    SILU_FAST_BODY(3, 4, 1, 2, 10, 12);  // k=5
    SILU_FAST_BODY(1, 2, 3, 4, 12, 14);  // k=6
    SILU_FAST_EPILOGUE();
}

#undef SILU_FAST_PROLOGUE
#undef SILU_FAST_BODY
#undef SILU_FAST_EPILOGUE
#endif  // !DISABLE_SFPLOADMACRO && SILU_BF16_FAST_ALLOW_DST_SPILL

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#if !defined(DISABLE_SFPLOADMACRO) && defined(SILU_BF16_FAST_ALLOW_DST_SPILL)
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_silu_bf16_fast_();
        return;
    }
    if constexpr (!is_fp32_dest_acc_en) {
        // ITERATIONS != 8: silu_init<..., false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG12 over the 2.0f that sfpu_reciprocal_iter (inside _sfpu_sigmoid_) reads; re-seed it.
        sfpu_reciprocal_init<false>();
    }
#endif
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so seed the reciprocal constant
    // (vConstFloatPrgm0 = 2.0f) exactly as the non-approx sigmoid_init does -- called directly rather than via
    // sigmoid_init<false, is_fp32_dest_acc_en>, which would also record calculate_sigmoid's fast replay body.
    sfpu_reciprocal_init<false>();
#if !defined(DISABLE_SFPLOADMACRO) && defined(SILU_BF16_FAST_ALLOW_DST_SPILL)
    // bf16: calculate_silu<false, 8> runs the fast kernel; program its state last so its LREG11..14 values win
    // (the ITERATIONS != 8 fallback re-seeds vConstFloatPrgm0).
    if constexpr (!is_fp32_dest_acc_en) {
        _init_silu_bf16_fast_();
    }
#endif
}

}  // namespace ckernel::sfpu
