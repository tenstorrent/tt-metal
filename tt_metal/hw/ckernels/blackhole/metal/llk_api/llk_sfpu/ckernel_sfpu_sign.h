// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Fast bf16 sign for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP (gate <= 0).
// Measured 90.1 cycles/tile vs 455.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// SFPU state programmed by the init: programmable constant LREG[11] = bf16 0x3C02, SFPLOADMACRO
// InstructionTemplate[0..2] (SFPCAST / SFP_STOCH_RND / SFPMUL) + LoadMacroConfig.Sequence[0] (macro 0) +
// Misc. No replay slots. bf16 DEST only (the final fp32->bf16 store truncation is what lands 127*k on
// exactly 1.0) -- gated on !is_fp32_dest_acc_en below; the sfpi loop stays as the fallback for every other
// configuration. Semantics match the fallback: sign(+/-0) = 0, sign(denormal) = +/-1, sign(+/-inf) = +/-1.
//
// One SFPLOADMACRO per 32-lane vector. Each SFPLOADMACRO performs the SFPLOAD and schedules the whole rest
// of the computation onto the four SFPU sub-units (round, simple, MAD, store), so the issue cost per vector
// is just the SFPLOADMACRO plus one SFPNOP spacer.
//
// Math (branch-free, denormal-safe -- all discrimination happens on raw bits):
//   1. t = SFPSTOCHRND IntInt (round unit): clamp sign-magnitude |bits(x)| to +/-127. Every nonzero bf16
//      input has fp32 bit-magnitude >= 0x10000 (bf16 << 16), so t = sign(x)|127 for ALL nonzero x
//      (denormals, inf, NaN included) and t = +0 for +/-0. Imm5=0 -> no shift, no rounding.
//   2. c = SFPCAST IntFloat (simple unit): t -> fp32 exactly: +/-127.0f or 0.
//   3. y = SFPMUL c * k (MAD unit), k = bf16 0x3C02 = 0.0079345703125f:
//      127*k = 1.00769... in [1.0, 1.0078125), 0*k = +0.
//   4. SFPSTORE with the load's Mod0 (SRCB->BF16): converts fp32->bf16 by truncating the mantissa toward
//      zero, so +/-1.00769 lands on exactly +/-1.0 and +0 stays 0. Store address = the SFPLOADMACRO's
//      address.
//
// Scheduling: all sub-unit delays use WaitForElapsedCycles, offsets relative to the SFPLOADMACRO cycle c:
// round c+1, simple c+2, MAD c+3 (2-cycle latency), store c+5. Loads rotate through L0..L3 so each vector's
// value lives well past its store. Consecutive SFPLOADMACROs must be >= 2 cycles apart (else the scheduled
// simple op of one macro and the round op of the next would land on the same cycle, which the HW only
// allows if one of them writes LReg[16]) -- the SFPNOP between them guarantees that; larger gaps (TRISC
// stalls, the wrapper's per-face code) are also collision-free.
#ifndef DISABLE_SFPLOADMACRO
inline void _init_sign_bf16_fast_() {
    // All lanes enabled so SFPCONFIG broadcasts reach every lane.
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPNOP;

    // L11 = k = bf16 0x3C02 (0.0079345703125f); 127*k truncates to 1.0 in bf16.
    TTI_SFPLOADI(0, 0 /*FLOATB*/, 0x3C02);
    TTI_SFPCONFIG(0, 11, 0);

    // Instruction templates, written via the backdoor: with LaneConfig.DISABLE_BACKDOOR_LOAD == 0 (the
    // init_sfpu default), an SFPU instruction whose VD is 12+i is captured into InstructionTemplate[i]
    // instead of executing. VD/VB-or-VC get overridden per SFPLOADMACRO.
    TTI_SFPCAST(0, 12, 0);                   // template[0]: CAST SM32->FP32 RNE, VC<-load
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 0xD);  // template[1]: IntInt clamp +/-127, RND_NEAREST, Imm5=0, VC<-load
    TTI_SFPMUL(11, 0, 9, 14, 0);             // template[2]: k * VB(<-load) + 0

    // Sequence[0], one byte per sub-unit (bit7: operand override VB=load (set) / VC=load (clear); bit6: dest
    // L16 (unused); bits3-5 delay; bits0-2 select: 3=builtin store, 4..7=template 0..3):
    //   simple = 0x0C: template[0] (CAST),     delay 1, VC=load, dest=load
    //   MAD    = 0x96: template[2] (MUL),      delay 2, VB=load, dest=load
    //   round  = 0x05: template[1] (STOCHRND), delay 0, VC=load, dest=load
    //   store  = 0x23: builtin SFPSTORE,       delay 4, VD=load, load's addr
    TTI_SFPLOADI(0, 10, 0x960C);  // low 16
    TTI_SFPLOADI(0, 8, 0x2305);   // high 16
    TTI_SFPCONFIG(0, 4, 0);       // LoadMacroConfig.Sequence[0] = LReg[0]

    // Misc: StoreMod0 unused (0), UsesLoadMod0ForStore = macro 0 (bit 4),
    // UnitDelayKind = WaitForElapsedCycles for all sub-units (bits 8-11 = 0).
    TTI_SFPCONFIG(0x010, 8, 1);
    TTI_SFPNOP;
}

// One face (8 dst vectors): dst offsets 0,2,...,14 (ADDR_MOD_7 = no increment).
// lreg_ind = (MacroIndex << 2) | VDLo -> macro 0, loads rotate L0..L3.
inline void _calculate_sign_bf16_fast_() {
    TTI_SFPLOADMACRO(0, 0, 7, 0);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(1, 0, 7, 2);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(2, 0, 7, 4);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(3, 0, 7, 6);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(0, 0, 7, 8);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(1, 0, 7, 10);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(2, 0, 7, 12);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(3, 0, 7, 14);
}
#endif

template <bool is_fp32_dest_acc_en>
inline void sign_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset) so this init is
    // self-contained (same convention as exp_init; sign uses only ADDR_MOD_7, no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en) {
        _init_sign_bf16_fast_();
    }
#endif
}

// exponent_size_8 is unused by both paths: the fast path discriminates on raw bits (bf16 always has an
// 8-bit exponent) and the sfpi fallback works on the converted fp32 value.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sign(const uint /*exponent_size_8*/) {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_sign_bf16_fast_();
        return;
    }
#endif
// All params are in FP16 format
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // copysgn stamps v's sign bit onto 1.0, which is exactly the v < 0 arm. Only the
        // zero case is left for a branch, so the v_elseif and its predicate-complement
        // disappear.
        sfpi::vFloat res = sfpi::copysgn(sfpi::vFloat(1.0f), v);
        // SFPSETCC is unspecified for -0.0 (VectorUnit.md), so a bare compare can miss it.
        v_if(sfpi::abs(v) == 0.0F) { res = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
