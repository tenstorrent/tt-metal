// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Seventh probe: every kernel family written with all five workarounds stacked
// (predicated store from a constant LReg, dest increment folded into the store
// addr_mode, nested v_if instead of &&, loop-invariant constant hoisted, and the
// integer compare commuted to the polarity SFPIADD's CC field can express).
#include <cstdint>
#include "shim.h"

using namespace sfpi;
constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;
constexpr int INFB = 0x7F800000;

extern "C" void w_gez_all() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 1.0f;
        v_if(v < 0.0f) {
            v_if(abs_bits != 0) { dst_reg[0] = 0.0f; }
            v_endif;
        }
        v_endif;
        v_if(as<vInt>(abs_bits) > inf_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

extern "C" void x_fp32_eq_all(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(as<vInt>(a) == as<vInt>(b)) { dst_reg[io * TS] = 1.0f; }
        v_endif;
        v_if(sum == 0.0f) { dst_reg[io * TS] = 1.0f; }
        v_endif;
        v_if(as<vInt>(sum) > inf_bits) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

extern "C" void y_fp32_le_all(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 1.0f;
        v_if(a > b) {
            v_if(sum != 0.0f) { dst_reg[io * TS] = 0.0f; }
            v_endif;
        }
        v_endif;
        v_if(as<vInt>(a) == as<vInt>(b)) { dst_reg[io * TS] = 1.0f; }
        v_endif;
        v_if(as<vInt>(sum) > inf_bits) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

// And the hypothetical: what le/ge would cost if a total-order compare existed
// and no inf-tie / denormal workarounds were needed. Modelled with the raw
// SFPGT builtin driving a lane mask, which is as close as sfpi lets us get.
extern "C" void z_le_with_total_order(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 1.0f;
        // total-order b < a, i.e. a > b, in one instruction
        vInt mask = as<vInt>(vFloat(__builtin_rvtt_sfpgt(b.get(), a.get(), 8)));
        v_if(mask != 0) {
            v_if(sum != 0.0f) { dst_reg[io * TS] = 0.0f; }
            v_endif;
        }
        v_endif;
        v_if(as<vInt>(sum) > inf_bits) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}
