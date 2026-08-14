// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Codegen probe: compile the sfpi bodies from PR #52932 in isolation and count
// the emitted SFPU instructions per DEST row.
#include <cstdint>
#include "shim.h"

using namespace sfpi;

constexpr int FP32_INF_BITS = 0x7F800000;

inline vFloat clear_sign(vFloat v) { return as<vFloat>(setsgn(as<vUInt>(v), 0)); }

// ---------------------------------------------------------------- comp.h: eqz
extern "C" void probe_eqz() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 0.0f;
        v_if(abs_bits == 0) { result = 1.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// ---------------------------------------------------------------- comp.h: ltz
extern "C" void probe_ltz() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 0.0f;
        v_if(v < 0.0f && abs_bits != 0) { result = 1.0f; }
        v_endif;
        v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// ---------------------------------------------------------------- comp.h: gez
extern "C" void probe_gez() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 1.0f;
        v_if(v < 0.0f && abs_bits != 0) { result = 0.0f; }
        v_endif;
        v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// ------------------------------------------------- binary_comp.h: fp32 eq
constexpr std::uint32_t TS = 32;
extern "C" void probe_fp32_eq(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = clear_sign(a) + clear_sign(b);
        vInt sum_bits = as<vInt>(sum);
        vFloat result = 0.0f;
        v_if(as<vInt>(a) == as<vInt>(b)) { result = 1.0f; }
        v_endif;
        v_if(sum == 0.0f) { result = 1.0f; }
        v_endif;
        v_if(sum_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;
        dst_reg[io * TS] = result;
        dst_reg++;
    }
}

// ------------------------------------------- binary_comp.h: fp32 strict (lt)
extern "C" void probe_fp32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = clear_sign(a) + clear_sign(b);
        vInt sum_bits = as<vInt>(sum);
        vFloat result = 0.0f;
        v_if(a < b && sum != 0.0f) { result = 1.0f; }
        v_endif;
        v_if(sum_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;
        dst_reg[io * TS] = result;
        dst_reg++;
    }
}

// --------------------------------------------- binary_comp.h: fp32 weak (le)
extern "C" void probe_fp32_le(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = clear_sign(a) + clear_sign(b);
        vInt sum_bits = as<vInt>(sum);
        vFloat result = 1.0f;
        v_if(a > b && sum != 0.0f) { result = 0.0f; }
        v_endif;
        v_if(as<vInt>(a) == as<vInt>(b)) { result = 1.0f; }
        v_endif;
        v_if(sum_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;
        dst_reg[io * TS] = result;
        dst_reg++;
    }
}

// -------------------------------------------- binary_comp.h: int32 lt (fold)
extern "C" void probe_int32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::I32>();
        vInt fold = a - as<vInt>(setsgn(as<vUInt>(b), 0));
        a = a ^ b;
        a = a | fold;
        a = a ^ b;
        dst_reg[io * TS].mode<sfpi::DataLayout::I32>() = as<vInt>(as<vUInt>(a) >> 31);
        dst_reg++;
    }
}

// ------------------------------- what the NAIVE sfpi int32 compare would cost
extern "C" void probe_int32_lt_naive(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::I32>();
        vInt result = 0;
        v_if(a < b) { result = 1; }
        v_endif;
        dst_reg[io * TS].mode<sfpi::DataLayout::I32>() = result;
        dst_reg++;
    }
}

// --------- the minimum possible: what a total-order compare primitive buys us
// A hypothetical sfpi "compare and produce a 0/-1 lane mask" on the raw bit
// patterns; modelled here on the uint16 path that main still keeps in raw TTI.
extern "C" void probe_ideal_mask(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::U32>();
        vUInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(a < b) { r = 1; }
        v_endif;
        dst_reg[io * TS].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}
