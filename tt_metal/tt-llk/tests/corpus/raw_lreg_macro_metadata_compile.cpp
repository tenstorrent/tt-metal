// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host syntax-only probe for the generated TT/TTI raw-LREG metadata shape.
// Select exactly one of TEST_WH, TEST_BH, or TEST_QSR on the command line.

#if defined(TEST_TARGET_EFFECT) && !defined(__riscv_xtt_sfprawlreg_effect)
#error "target check requires a Tensix compiler implementing raw-LREG effects"
#endif

namespace ckernel
{
inline volatile unsigned instrn_buffer[1];
}

#if !defined(TEST_DEFAULT_FALLBACK)
#if !defined(TEST_FALLBACK)
inline void raw_lreg_marker(unsigned, unsigned) {}
#define TT_LLK_SFPRAWLREG_EFFECT(read_mask, write_mask) raw_lreg_marker((read_mask), (write_mask))
#endif
#endif

#if defined(TEST_WH)
#include "../../tt_llk_wormhole_b0/common/inc/ckernel_ops.h"
#elif defined(TEST_BH)
#include "../../tt_llk_blackhole/common/inc/ckernel_ops.h"
#elif defined(TEST_QSR)
#include "../../tt_llk_quasar/common/inc/ckernel_ops.h"
#else
#error "select a test architecture"
#endif

constexpr bool effect_is(unsigned word, unsigned read_mask, unsigned write_mask)
{
    return ckernel::raw_lreg_effect::read(word) == read_mask && ckernel::raw_lreg_effect::write(word) == write_mask;
}

constexpr unsigned sfpadd_1234 = TT_OP_SFPADD(1, 2, 3, 4, 0);
static_assert(effect_is(sfpadd_1234, 0x0eu, 0x10u));
// A lane-merge compiler owns destination preservation. Older effect compilers
// require the old destination explicitly even for write-only instructions.
#if defined(__riscv_xtt_sfprawlreg_merge)
static_assert(ckernel::raw_lreg_effect::inputs(sfpadd_1234) == 0x0eu);
static_assert(ckernel::raw_lreg_effect::inputs(TT_OP_SFPLOADI(3, 0, 0)) == 0x00u);
#else
static_assert(ckernel::raw_lreg_effect::inputs(sfpadd_1234) == 0x1eu);
static_assert(ckernel::raw_lreg_effect::inputs(TT_OP_SFPLOADI(3, 0, 0)) == 0x08u);
#endif
static_assert(effect_is(TT_OP_SFPMAD(1, 2, 3, 4, 4), 0xffu, 0x10u));
static_assert(effect_is(TT_OP_SFPMAD(1, 2, 3, 4, 8), 0x8eu, 0xffu));
static_assert(effect_is(TT_OP_SFPADDI(0, 3, 8), 0x88u, 0xffu));
static_assert(effect_is(TT_OP_SFPADDI(0, 3, 4), 0xffu, 0x08u));
static_assert(effect_is(TT_OP_SFPMULI(0, 3, 4), 0xffu, 0x08u));
static_assert(effect_is(TT_OP_SFPMULI(0, 3, 12), 0xffu, 0xffu));
static_assert(effect_is(TT_OP_SFPLOADI(3, 8, 0), 0x08u, 0x08u));
static_assert(effect_is(TT_OP_SFPSWAP(0, 1, 2, 0), 0x66u, 0x66u));
static_assert(effect_is(TT_OP_SFPSHFT2(1, 5, 3, 0), 0x0eu, 0x0fu));
static_assert(effect_is(TT_OP_SFPSHFT2(1, 5, 3, 1), 0x0fu, 0x0fu));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 2)) == 0x2eu);
static_assert(effect_is(TT_OP_SFPSHFT2(1, 5, 3, 3), 0x20u, 0x08u));
static_assert(effect_is(TT_OP_SFPSHFT2(1, 5, 3, 4), 0x20u, 0x08u));
static_assert(effect_is(TT_OP_SFPSHFT2(1, 5, 3, 5), 0x2au, 0x08u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 6)) == 0x0au);
// Production shape: TTI_SFPSHFT2(LREG0, LREG13, LREG2, 5) shifts LREG0.
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(0, 13, 2, 5)) & 0x01u);
static_assert(ckernel::raw_lreg_effect::supported(TT_OP(0x93, 0)));
static_assert(effect_is(TT_OP(0x93, 0), 0xffu, 0xffu));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 2, 1)) == 0x01u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 4, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 4, 0)) == 0x01u);
// Opcodes outside this architecture's SFPU range fail the gate, and an opcode
// the decoder does not model reports every LREG rather than none.
#if defined(TEST_WH)
static_assert(ckernel::raw_lreg_effect::supported(TT_OP(0x95, 0)));
static_assert(!ckernel::raw_lreg_effect::supported(TT_OP(0x96, 0)));
#else
static_assert(ckernel::raw_lreg_effect::supported(TT_OP(0x99, 0)));
#endif
static_assert(!ckernel::raw_lreg_effect::supported(TT_OP(0x6f, 0)));
static_assert(!ckernel::raw_lreg_effect::supported(TT_OP(0x9a, 0)));
static_assert(effect_is(TT_OP(0x9a, 0), 0xffu, 0xffu));
static_assert(effect_is(TT_OP(0x8f, 0), 0x00u, 0x00u));
static_assert(ckernel::raw_lreg_effect::write(TT_OP(0x72, 0)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP(0x91, 0)) == 0x00u);
#if defined(TEST_QSR)
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(1, 2, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 8)) == 0x00u);
static_assert(effect_is(TT_OP_SFPLOAD(2, 14, 0, 0, 0), 0x04u, 0x44u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSTORE(5, 0, 0, 0, 0)) == 0x20u);
static_assert(effect_is(TT_OP_SFPLUT(3, 0), 0x0fu, 0x08u));
static_assert(effect_is(TT_OP_SFPLUT(3, 8), 0x8fu, 0xffu));
static_assert(effect_is(TT_OP_SFPAND(2, 3), 0x0cu, 0x08u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 4)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 5)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 1)) == 0x08u);
static_assert(effect_is(TT_OP_SFPTRANSP, 0xffu, 0xffu));
static_assert(effect_is(TT_OP_SFPMUL24(1, 2, 4, 0), 0x06u, 0x10u));
static_assert(effect_is(TT_OP_SFPMUL24(1, 2, 4, 8), 0x86u, 0xffu));
static_assert(effect_is(TT_OP_SFPNONLINEAR(2, 3, 0), 0x04u, 0x08u));
#else
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(1, 2, 0, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0, 8)) == 0x00u);
static_assert(effect_is(TT_OP_SFPLOAD(2, 14, 0, 0), 0x04u, 0x44u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSTORE(5, 0, 0, 0)) == 0x20u);
static_assert(effect_is(TT_OP_SFPLUT(3, 0, 0), 0x0fu, 0x08u));
static_assert(effect_is(TT_OP_SFPLUT(3, 8, 0), 0x8fu, 0xffu));
static_assert(effect_is(TT_OP_SFPAND(0, 2, 3, 0), 0x0cu, 0x08u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 4)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 5)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 1)) == 0x08u);
#if defined(TEST_BH)
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(5, 2, 3, 1)) == 0x24u);
#else
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(5, 2, 3, 1)) == 0x0cu);
#endif
static_assert(effect_is(TT_OP_SFPTRANSP(0, 0, 0, 0), 0xffu, 0xffu));
#if defined(TEST_BH)
static_assert(effect_is(TT_OP_SFPMUL24(1, 2, 7, 4, 0), 0x86u, 0x10u));
static_assert(effect_is(TT_OP_SFPMUL24(1, 2, 3, 4, 4), 0xffu, 0x10u));
static_assert(effect_is(TT_OP_SFPARECIP(5, 2, 3, 1), 0x24u, 0x08u));
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPARECIP(5, 2, 3, 2)) == 0x04u);
#endif
#endif
static_assert(effect_is(TT_OP_SFPLUTFP32(4, 0), 0x7fu, 0x10u));
static_assert(effect_is(TT_OP_SFPLUTFP32(4, 2), 0x7fu, 0x10u));
static_assert(effect_is(TT_OP_SFPLUTFP32(4, 3), 0x7fu, 0x10u));
static_assert(effect_is(TT_OP_SFPLUTFP32(4, 10), 0x8fu, 0xffu));

// WH/BH ISA SFPLUTFP32 functional model: table selection uses bits 0/1,
// sign retention uses bit 2, and indirect addressing uses bit 3. Packed
// three-entry coefficients need L0..L3; other tables also need L4..L6.
constexpr bool lut_modifier_masks()
{
    constexpr unsigned expected_reads[16] = {
        0x7f, 0, 0x7f, 0x7f, 0x7f, 0, 0x7f, 0x7f,
        0, 0, 0x8f, 0, 0, 0, 0x8f, 0
    };
    for (unsigned mode = 0; mode != 16; ++mode) {
        const unsigned word = TT_OP_SFPLUTFP32(4, mode);
        if (ckernel::raw_lreg_effect::supported(word) != (expected_reads[mode] != 0)) return false;
        if (!expected_reads[mode]) continue;
        if (ckernel::raw_lreg_effect::read(word) != expected_reads[mode]) return false;
        if (ckernel::raw_lreg_effect::write(word) != (mode >= 8 ? 0xffu : 0x10u)) return false;
    }
    return true;
}
static_assert(lut_modifier_masks());

// REPLAY playback (load_mode 0) touches every LREG; recording touches none.
static_assert(ckernel::raw_lreg_effect::replay_effect(TT_OP(0x04, 0)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::replay_effect(TT_OP(0x04, 1)) == 0x00u);

// Compile the explicit template and known-macro paths on every architecture.
// They preserve the statement-expression contract of ordinary TT/TTI macros.
void known_macro_issue()
{
    TT_LLK_SFPU_ISSUE_TTI_TEMPLATE(TT_OP_SFPMAD(0, 0, 0, 13, 4));
    TT_LLK_SFPU_ISSUE_TT_MACRO(TT_OP(0x93, 0), 0x80u, 0x03u);
}

// Wrappers decode effects from a word with address and immediate operands
// zeroed; those operands must never change the decoded masks.
constexpr bool same_effect(unsigned issued, unsigned lreg_word)
{
    return ckernel::raw_lreg_effect::inputs(issued) == ckernel::raw_lreg_effect::inputs(lreg_word) &&
           ckernel::raw_lreg_effect::write(issued) == ckernel::raw_lreg_effect::write(lreg_word);
}
constexpr bool lreg_word_operands_ignored()
{
    for (unsigned mode = 0; mode != 16; ++mode) {
        for (unsigned lreg = 0; lreg != 8; ++lreg) {
#if defined(TEST_QSR)
            if (!same_effect(TT_OP_SFPLOAD(lreg, mode, 3, 1, 0x2a5), TT_OP_SFPLOAD(lreg, mode, 0, 0, 0))) return false;
            if (!same_effect(TT_OP_SFPSTORE(lreg, mode, 3, 1, 0x2a5), TT_OP_SFPSTORE(lreg, mode, 0, 0, 0))) return false;
#else
            if (!same_effect(TT_OP_SFPLOAD(lreg, mode, 3, 0x1abc), TT_OP_SFPLOAD(lreg, mode, 0, 0))) return false;
            if (!same_effect(TT_OP_SFPSTORE(lreg, mode, 3, 0x1abc), TT_OP_SFPSTORE(lreg, mode, 0, 0))) return false;
            if (!same_effect(TT_OP_SFPLUT(lreg, mode, 0xbeef), TT_OP_SFPLUT(lreg, mode, 0))) return false;
#endif
            if (!same_effect(TT_OP_SFPLOADI(lreg, mode, 0xbeef), TT_OP_SFPLOADI(lreg, mode, 0))) return false;
            if (!same_effect(TT_OP_SFPADDI(0xbeef, lreg, mode), TT_OP_SFPADDI(0, lreg, mode))) return false;
            if (!same_effect(TT_OP_SFPMULI(0xbeef, lreg, mode), TT_OP_SFPMULI(0, lreg, mode))) return false;
        }
        for (unsigned dest = 0; dest != 16; ++dest) {
            if (!same_effect(TT_OP_SFPCONFIG(0xbeef, dest, mode), TT_OP_SFPCONFIG(0, dest, mode))) return false;
        }
    }
    return true;
}
static_assert(lreg_word_operands_ignored());

#ifdef TEST_LUT_MODES
void lut_modifier_issue()
{
    TTI_SFPLUTFP32(4, 0);
    TTI_SFPLUTFP32(4, 2);
    TTI_SFPLUTFP32(4, 3);
    TTI_SFPLUTFP32(4, 4);
    TTI_SFPLUTFP32(4, 6);
    TTI_SFPLUTFP32(4, 7);
    TTI_SFPLUTFP32(4, 10);
    TTI_SFPLUTFP32(4, 14);
}
#endif

#ifdef TEST_EXPRESSION_POSITION
void expression_position_issue(bool select)
{
    select ? TTI_SFPADD(1, 2, 3, 4, 0) : TTI_SFPADD(1, 2, 3, 5, 0);
    (void)TT_SFPLOADI(4, 0, 0), TT_SFPLOADI(5, 0, 0);
}
#endif

#ifdef TEST_RESERVED_LUT
void reserved_lut_modifier_must_refuse()
{
    TTI_SFPLUTFP32(4, 8);
}
#endif

void tt_mmio_constant_lreg()
{
#if defined(TEST_QSR)
    TT_SFPLOAD(3, 0, 0, 0, 0);
    TT_SFPLOADMACRO(0, 1, 0, 0, 0, 0, 0);
#else
    TT_SFPLOAD(3, 0, 0, 0);
    TT_SFPLOADMACRO(1, 0, 0, 0);
#endif
    TT_SFPLOADI(4, 0, 0);
    TT_SFPADD(1, 2, 3, 4, 0);
    TT_SFPSWAP(0, 1, 2, 0);
#if defined(TEST_QSR)
    TT_SFPTRANSP;
#else
    TT_SFPTRANSP(0, 0, 0, 0);
#endif
}

void tt_mmio_runtime_lreg(unsigned lreg)
{
#if defined(TEST_QSR)
    TT_SFPLOAD(lreg, 0, 0, 0, 0);
#else
    TT_SFPLOAD(lreg, 0, 0, 0);
#endif
}

// The address is a runtime value, but the LREG operands are constants, so the
// effect stays exact at every optimization level.
void tt_mmio_runtime_address(unsigned address)
{
#if defined(TEST_QSR)
    TT_SFPLOAD(3, 0, 0, 0, address);
#else
    TT_SFPLOAD(3, 0, 0, address);
#endif
}

// LREG operands become constant only after inlining: exact when optimizing,
// all LREGs otherwise. Forced inline keeps the -O0 helper-call scan meaningful.
__attribute__((always_inline)) inline void tt_mmio_load_into(unsigned lreg)
{
#if defined(TEST_QSR)
    TT_SFPLOAD(lreg, 0, 0, 0, 0);
#else
    TT_SFPLOAD(lreg, 0, 0, 0);
#endif
}
void tt_mmio_inlined_lreg() { tt_mmio_load_into(2); }

void tti_asm_constant_lreg()
{
#if defined(TEST_QSR)
    TTI_SFPLOAD(5, 0, 0, 0, 0);
    TTI_SFPLOADMACRO(0, 1, 0, 0, 0, 0, 0);
#else
    TTI_SFPLOAD(5, 0, 0, 0);
    TTI_SFPLOADMACRO(1, 0, 0, 0);
#endif
    TTI_SFPLOADI(6, 0, 0);
    TTI_SFPADD(1, 2, 3, 4, 0);
    TTI_SFPSHFT2(0, 1, 2, 3);
    TTI_SFPLUTFP32(2, 0);
}
