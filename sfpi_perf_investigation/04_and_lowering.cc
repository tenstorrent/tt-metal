// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fourth probe: what breaks the cheap chained-SETCC form for `&&`?
#include <cstdint>
#include "shim.h"

using namespace sfpi;
constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;

// P) three-term AND where every term compares against a SMALL immediate that
//    fits SFPIADD's 12-bit field (so no SFPLOADI is needed mid-chain).
extern "C" void p_three_term_small_imm(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vInt ai = as<vInt>(a);
        dst_reg[io * TS] = 0.0f;
        v_if(a < b && ai != 0 && ai < 100) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// Q) same three terms, but the wide immediate is pre-loaded into a vInt so the
//    condition itself needs no SFPLOADI.
extern "C" void q_three_term_preloaded(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b && sum != 0.0f && as<vInt>(sum) <= inf_bits) {
            dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f;
        }
        v_endif;
    }
}

// R) the same predicate expressed as nested v_if instead of a single &&.
extern "C" void r_nested_vif(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b) {
            v_if(sum != 0.0f) {
                v_if(as<vInt>(sum) <= inf_bits) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}
