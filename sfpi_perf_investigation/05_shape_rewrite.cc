// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fifth probe: rewrite every kernel family in the *shape* the raw-TTI original
// used (predicated store, dest-increment folded into the store's addr_mode,
// nested v_if instead of &&) to establish the best sfpi can do today.
#include <cstdint>
#include "shim.h"

using namespace sfpi;
constexpr std::uint32_t TS = 32;
constexpr int AM6 = 6;
constexpr int INFB = 0x7F800000;

// ============================== comp.h ==============================
extern "C" void best_eqz() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits == 0) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

extern "C" void best_ltz() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(v < 0.0f) {
            v_if(abs_bits != 0) {
                v_if(abs_bits <= inf_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

extern "C" void best_gez() {
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
        v_if(abs_bits > inf_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 0.0f; }
        v_endif;
    }
}

// ============================ binary_comp.h ============================
extern "C" void best_fp32_eq(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
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

extern "C" void best_fp32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = INFB;
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

extern "C" void best_fp32_le(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
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

extern "C" void best_int32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::I32>();
        vInt fold = a - as<vInt>(setsgn(as<vUInt>(b), 0));
        a = a ^ b;
        a = a | fold;
        a = a ^ b;
        dst_reg[io * TS].mode<sfpi::DataLayout::I32>(AM6) = as<vInt>(as<vUInt>(a) >> 31);
    }
}
