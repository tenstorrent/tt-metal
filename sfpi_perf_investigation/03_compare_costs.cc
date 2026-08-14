// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Third probe: is a total-order (sign-magnitude) compare reachable from v_if,
// and what does the float relational operator actually cost?
#include <cstdint>
#include "shim.h"

using namespace sfpi;

constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;

// H) sign-magnitude compare through the vSMag type + explicit vBool Cond.
extern "C" void h_smag_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vSMag a = as<vSMag>(vUInt(dst_reg[i0 * TS].mode<sfpi::DataLayout::U32>()));
        vSMag b = as<vSMag>(vUInt(dst_reg[i1 * TS].mode<sfpi::DataLayout::U32>()));
        dst_reg[io * TS] = 0.0f;
        v_if(vBool(vBool::LT, a, b)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// I) isolated cost of the float relational operator.
extern "C" void i_float_lt_only(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        dst_reg[io * TS] = 0.0f;
        v_if(a < b) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// J) isolated cost of the float weak relational operator (the > / >= family).
extern "C" void j_float_gt_only(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        dst_reg[io * TS] = 1.0f;
        v_if(a > b) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

// K) two-term AND with a predicated store (does it stay in the cheap chained
//    SETCC form, or does it fall back to PUSHC/POPC?)
extern "C" void k_two_term_and(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b && sum != 0.0f) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// L) three-term AND with a register assignment instead of a predicated store.
extern "C" void l_three_term_and_reg(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        vFloat result = 0.0f;
        v_if(a < b && sum != 0.0f && as<vInt>(sum) <= 0x7F800000) { result = 1.0f; }
        v_endif;
        dst_reg[io * TS] = result;
        dst_reg++;
    }
}

// M) unsigned 32-bit compare: is it exact, or does it also go through a
//    signed subtract? (the uint ordering path main still keeps in raw TTI)
extern "C" void m_uint32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::U32>();
        vUInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(a < b) { r = 1; }
        v_endif;
        dst_reg[io * TS].mode<sfpi::DataLayout::U32>(AM6) = r;
    }
}

// N) the SM32 dest layout: does loading DEST as sign-magnitude give a
//    total-order compare for free?
extern "C" void n_sm32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vSMag a = dst_reg[i0 * TS].mode<sfpi::DataLayout::SM32>();
        vSMag b = dst_reg[i1 * TS].mode<sfpi::DataLayout::SM32>();
        dst_reg[io * TS] = 0.0f;
        v_if(vBool(vBool::LT, a, b)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
