// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Eighth probe: commute the remaining NaN tests too (inf < x instead of x > inf)
// so every integer compare hits a CC polarity SFPIADD can fuse.
#include <cstdint>
#include "shim.h"

using namespace sfpi;
constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;
constexpr int INFB = 0x7F800000;

extern "C" void best2_gez() {
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
        v_if(inf_bits < abs_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

extern "C" void best2_fp32_eq(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
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
        v_if(inf_bits < as<vInt>(sum)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

extern "C" void best2_fp32_le(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
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
        v_if(inf_bits < as<vInt>(sum)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

// And le/ge if a total-order compare existed: no inf-tie clause needed, and the
// compare itself is one instruction.
extern "C" void ideal_fp32_le(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 1.0f;
        vInt gt = as<vInt>(vFloat(__builtin_rvtt_sfpgt(b.get(), a.get(), 8)));
        v_if(gt != 0) {
            v_if(sum != 0.0f) { dst_reg[io * TS] = 0.0f; }
            v_endif;
        }
        v_endif;
        v_if(inf_bits < as<vInt>(sum)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

extern "C" void ideal_fp32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        vInt lt = as<vInt>(vFloat(__builtin_rvtt_sfpgt(a.get(), b.get(), 8)));
        v_if(lt != 0) {
            v_if(sum != 0.0f) {
                v_if(inf_bits >= as<vInt>(sum)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}
