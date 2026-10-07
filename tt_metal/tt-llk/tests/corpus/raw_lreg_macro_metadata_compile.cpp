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

static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPADD(1, 2, 3, 4, 0)) == 0x0eu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPADD(1, 2, 3, 4, 0)) == 0x10u);
// The macro contract includes the old destination for inactive lanes, even
// when the opcode has no explicit destination read.
static_assert(ckernel::raw_lreg_effect::inputs(TT_OP_SFPADD(1, 2, 3, 4, 0)) == 0x1eu);
static_assert(ckernel::raw_lreg_effect::inputs(TT_OP_SFPLOADI(3, 0, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMAD(1, 2, 3, 4, 4)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMAD(1, 2, 3, 4, 4)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMAD(1, 2, 3, 4, 8)) == 0x8eu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMAD(1, 2, 3, 4, 8)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPADDI(0, 3, 8)) == 0x88u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPADDI(0, 3, 8)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLOADI(3, 8, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLOADI(3, 8, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSWAP(0, 1, 2, 0)) == 0x66u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSWAP(0, 1, 2, 0)) == 0x66u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 0)) == 0x0eu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSHFT2(1, 5, 3, 0)) == 0x0fu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 1)) == 0x0fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSHFT2(1, 5, 3, 1)) == 0x0fu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 2)) == 0x2eu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 3)) == 0x20u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSHFT2(1, 5, 3, 3)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 4)) == 0x20u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSHFT2(1, 5, 3, 4)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 5)) == 0x22u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPSHFT2(1, 5, 3, 5)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT2(1, 5, 3, 6)) == 0x02u);
static_assert(ckernel::raw_lreg_effect::supported(TT_OP(0x93, 0)));
static_assert(ckernel::raw_lreg_effect::read(TT_OP(0x93, 0)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP(0x93, 0)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 2, 1)) == 0x01u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 4, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPCONFIG(0, 4, 0)) == 0x01u);
#if defined(TEST_QSR)
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(1, 2, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 8)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLOAD(2, 14, 0, 0, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLOAD(2, 14, 0, 0, 0)) == 0x44u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSTORE(5, 0, 0, 0, 0)) == 0x20u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUT(3, 0)) == 0x0fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUT(3, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUT(3, 8)) == 0x8fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUT(3, 8)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(2, 3)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPAND(2, 3)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 4)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 5)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPTRANSP) == 0xffu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPTRANSP) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMUL24(1, 2, 4, 0)) == 0x06u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMUL24(1, 2, 4, 0)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMUL24(1, 2, 4, 8)) == 0x86u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMUL24(1, 2, 4, 8)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPNONLINEAR(2, 3, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPNONLINEAR(2, 3, 0)) == 0x08u);
#else
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(1, 2, 0, 1)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSETCC(0, 2, 0, 8)) == 0x00u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLOAD(2, 14, 0, 0)) == 0x04u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLOAD(2, 14, 0, 0)) == 0x44u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSTORE(5, 0, 0, 0)) == 0x20u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUT(3, 0, 0)) == 0x0fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUT(3, 0, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUT(3, 8, 0)) == 0x8fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUT(3, 8, 0)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(0, 2, 3, 0)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPAND(0, 2, 3, 0)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 4)) == 0x0cu);
#if defined(TEST_BH)
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(5, 2, 3, 1)) == 0x24u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 5)) == 0x0cu);
#else
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPAND(5, 2, 3, 1)) == 0x0cu);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPSHFT(0, 2, 3, 5)) == 0x08u);
#endif
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPTRANSP(0, 0, 0, 0)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPTRANSP(0, 0, 0, 0)) == 0xffu);
#if defined(TEST_BH)
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMUL24(1, 2, 7, 4, 0)) == 0x86u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMUL24(1, 2, 7, 4, 0)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPMUL24(1, 2, 3, 4, 4)) == 0xffu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPMUL24(1, 2, 3, 4, 4)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPARECIP(5, 2, 3, 1)) == 0x24u);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPARECIP(5, 2, 3, 1)) == 0x08u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPARECIP(5, 2, 3, 2)) == 0x04u);
#endif
#endif
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUTFP32(4, 0)) == 0x7fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUTFP32(4, 0)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUTFP32(4, 2)) == 0x7fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUTFP32(4, 2)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUTFP32(4, 3)) == 0x7fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUTFP32(4, 3)) == 0x10u);
static_assert(ckernel::raw_lreg_effect::read(TT_OP_SFPLUTFP32(4, 10)) == 0x8fu);
static_assert(ckernel::raw_lreg_effect::write(TT_OP_SFPLUTFP32(4, 10)) == 0xffu);

// WH/BH ISA SFPLUTFP32 functional model: table selection uses bits 0/1,
// sign retention uses bit 2, and indirect addressing uses bit 3. Packed
// three-entry coefficients need L0..L3; other tables also need L4..L6.
constexpr bool lut_modifier_masks()
{
    constexpr unsigned expected_reads[16] = {
        0x7f, 0, 0x7f, 0x7f, 0x7f, 0, 0x7f, 0x7f,
        0xff, 0, 0x8f, 0x8f, 0xff, 0, 0x8f, 0x8f
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

#ifdef TEST_LUT_MODES
void lut_modifier_issue()
{
    TTI_SFPLUTFP32(4, 0);
    TTI_SFPLUTFP32(4, 2);
    TTI_SFPLUTFP32(4, 3);
    TTI_SFPLUTFP32(4, 4);
    TTI_SFPLUTFP32(4, 6);
    TTI_SFPLUTFP32(4, 7);
    TTI_SFPLUTFP32(4, 8);
    TTI_SFPLUTFP32(4, 10);
    TTI_SFPLUTFP32(4, 11);
    TTI_SFPLUTFP32(4, 12);
    TTI_SFPLUTFP32(4, 14);
    TTI_SFPLUTFP32(4, 15);
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
