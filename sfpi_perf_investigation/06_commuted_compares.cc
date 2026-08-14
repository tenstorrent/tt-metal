// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sixth probe: does commuting the comparison avoid the SFPCOMPC, and is the
// integer compare overflow visible in codegen for both vInt and vUInt?
#include <cstdint>
#include "shim.h"

using namespace sfpi;
constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;
constexpr int INFB = 0x7F800000;

// S) `abs <= inf` (what reads naturally)
extern "C" void s_le_form() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits <= inf_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// T) the same predicate commuted to `inf >= abs`, which is the polarity the
//    SFPIADD condition-code field can express directly.
extern "C" void t_ge_form() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(inf_bits >= abs_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// U) ltz with every workaround stacked, including the commuted NaN test.
extern "C" void u_ltz_all_workarounds() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(v < 0.0f) {
            v_if(abs_bits != 0) {
                v_if(inf_bits >= abs_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// V) fp32 strict lt with every workaround stacked.
extern "C" void v_fp32_lt_all_workarounds(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b) {
            v_if(sum != 0.0f) {
                v_if(inf_bits >= as<vInt>(sum)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}
