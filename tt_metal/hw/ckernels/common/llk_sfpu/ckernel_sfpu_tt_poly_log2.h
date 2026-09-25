// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly {
#if defined(ARCH_WORMHOLE)
constexpr uint32_t kLog2Hold = ADDR_MOD_3;
constexpr uint32_t kLog2Advance = ADDR_MOD_2;
constexpr bool kLog2RawPartition = true;
#elif defined(ARCH_BLACKHOLE)
constexpr uint32_t kLog2Hold = ADDR_MOD_7;
constexpr uint32_t kLog2Advance = ADDR_MOD_6;
constexpr bool kLog2RawPartition = false;
#else
#error "logarithm replay requires BH or WH"
#endif

template <typename Config>
constexpr bool log2_contract() {
    return Config::kDegree == 6u && Config::kCoefficientBits[0] == 0u && Config::kTerminalSlots == 7u &&
           Config::kInputMinNormalBits == 0x00800000u && Config::kRawPartition == kLog2RawPartition &&
           Config::kBodySlots == (Config::kScaleBits == 0x3f800000u ? 30u : 32u);
}

template <typename Config>
inline void init_log2() {
    static_assert(log2_contract<Config>());
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kScaleBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kCoefficientBits[6]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kCoefficientBits[5]);
    // The upstream callback owns face traversal. Advance one vector per store.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

template <typename Config, int Iterations = 8>
inline void calculate_log2() {
    static_assert(log2_contract<Config>());
    static_assert(Iterations > 0);
    // Keep pins local to the callback: upstream traversal may use local registers.
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[4] >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[4] & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[3] >> 16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[3] & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[2] >> 16);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[2] & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, kLog2Hold, 0);  // x
#if defined(ARCH_WORMHOLE)
    // Ten-slot prelude including the float LOAD above, replacing the
    // plain LOAD/NOP/EXEXP (net +7). Physical U16 retains the original
    // sign in bit15 and exponent in bits0..7. Masked zero is exactly
    // positive zero/subnormal; 1..254 is exactly positive normal.
    // Negative zero/subnormals therefore retain the declared +Inf.
    // The masked value already IS the core's biased exponent, so no
    // EXEXP is needed. Only L0..L3 are scratch; L4..L7 retain pins.
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_UINT16, kLog2Hold, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
    // WH only supports the destructive two-register AND (mod1=0).
    // The mask load also supplies the raw LOAD's required gap.
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0xff80);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(0xF01, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
#else
    // Eight instructions surround the already-required SFPEXEXP and
    // replace the plain body's leading NOP, so the route costs seven
    // replay slots. L3 defaults to qNaN. The first CC window combines
    // x>=0 with x<target MIN_NORMAL, overwriting +zero and every
    // positive BF16 subnormal with -Inf. The threshold comes from the
    // verifier-bound BF16+DAZ target lattice, not an activation or
    // max-subnormal constant. After closing that action window, the
    // ordered MIN_NORMAL<=x predicate plus biased exponent<255 opens
    // the unchanged core on exactly positive-normal-finite x. L0 is
    // dead until ABS below, so both threshold and exponent scratch are
    // overwritten normally. NODEBIAS SFPEXEXP and the selected core's
    // original L1-=127 remain in their original arithmetic order.
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0x7FC0);  // qNaN default / load gap
    TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG1,
                 sfpi::SFPEXEXP_MOD1_NODEBIAS);  // selected-core biased exponent
    TTI_SFPLOADI(p_sfpu::LREG0, 0,
                 Config::kInputMinNormalBits >> 16);  // BF16 MIN_NORMAL
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPGT(0, p_sfpu::LREG2, p_sfpu::LREG0, 1);  // x < MIN_NORMAL
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0xFF80);         // positive exponent-zero -> -Inf
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPLE(0, p_sfpu::LREG2, p_sfpu::LREG0, 1);  // MIN_NORMAL <= x
    TTI_SFPIADD(
        0xF01,
        p_sfpu::LREG1,
        p_sfpu::LREG0,
        sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);  // biased exponent - 255 < 0
#endif
    TTI_SFPSETEXP(127, p_sfpu::LREG2, p_sfpu::LREG2, 1);  // m (imm exponent)
    TTI_SFPIADD(0xF81, p_sfpu::LREG1, p_sfpu::LREG1, 5);  // e -= 127 (imm, no CC)
    // Operand order mirrors the sfpi-emitted word exactly
    // (a=LCONST_1, b=m): 1.0*m + (-1.0) — bit-identical either way.
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        p_sfpu::LREG2,
        p_sfpu::LCONST_neg1,
        p_sfpu::LREG2,
        0);                                          // u = m + (-1.0) (hw const L11; MAD-unit producer)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // |e| (int); u-gap filler
    TTI_SFPMAD(p_sfpu::LREG13, p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG3,
               0);                                 // h = c[D]*u + c[D-1]
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);  // float(|e|); h-gap filler
    TTI_SFPMAD(
        p_sfpu::LREG3,
        p_sfpu::LREG2,
        (Config::kDegree >= 3) ? p_sfpu::LREG7 : p_sfpu::LCONST_0,
        p_sfpu::LREG3,
        0);                                             // rung 1: + c[D-2] (or c0 == 0 -> hw zero at DEG 2)
    TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // e_f={sgn e,mag |e|}
    if constexpr (Config::kDegree >= 3) {
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 4) ? p_sfpu::LREG6 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 2 (gapped by the SETSGN above)
    }
    if constexpr (Config::kDegree >= 4) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 5) ? p_sfpu::LREG5 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 3
    }
    if constexpr (Config::kDegree >= 5) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 6) ? p_sfpu::LREG4 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 4
    }
    if constexpr (Config::kDegree >= 6) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 5: + c0 == 0 (hw zero, gate-proved)
    }
    TTI_SFPNOP;
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        p_sfpu::LREG1,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        0);  // e_float + h (a=LCONST_1 form, mirrors the sfpi sfpadd)
    if constexpr (Config::kScaleBits != 0x3f800000u) {
        TTI_SFPNOP;
        TTI_SFPMUL(
            p_sfpu::LREG3,
            p_sfpu::LREG12,
            p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // * LOG_HW_SCALE (prgm0; exact-0.0 addend == sfpmul)
    }
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);        // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kLog2Advance, 0);  // dest += 2
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_LOG2_REPLAY_V1 1
