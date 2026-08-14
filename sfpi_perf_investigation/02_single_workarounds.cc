// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Second codegen probe: test which sfpi features would close each part of the gap.
#include <cstdint>
#include "shim.h"

using namespace sfpi;

constexpr int FP32_INF_BITS = 0x7F800000;
constexpr std::uint32_t TS = 32;
// ADDR_MOD_6 is the mod the *_init() functions program with dest.incr = 2,
// i.e. exactly what the raw-TTI kernels fold their final store into.
constexpr int AM6 = 6;

// A) int32 fold, but folding the DEST advance into the store's addr_mode
//    instead of a separate dst_reg++ / TTINCRWC.
extern "C" void a_int32_lt_addrmod(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
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

// B) ltz with the +inf bit pattern hoisted into a vInt declared outside the loop.
extern "C" void b_ltz_hoisted() {
    vInt inf_bits = FP32_INF_BITS;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 0.0f;
        v_if(v < 0.0f && abs_bits != 0) { result = 1.0f; }
        v_endif;
        v_if(abs_bits > inf_bits) { result = 0.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// C) eqz written as a predicated *store* (the raw-TTI shape) instead of a
//    predicated register assignment followed by an unconditional store.
extern "C" void c_eqz_predicated_store() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits == 0) { dst_reg[0] = 1.0f; }
        v_endif;
        dst_reg++;
    }
}

// D) can the total-order SFPGT builtin be reached at all?
extern "C" void d_raw_sfpgt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        // SFPGT_MOD1_SET_VD == 8: vd = (a > b) ? -1 : 0, total order on the bits.
        vUInt mask = as<vUInt>(vFloat(__builtin_rvtt_sfpgt(a.get(), b.get(), 8)));
        dst_reg[io * TS].mode<sfpi::DataLayout::U32>() = mask >> 31;
        dst_reg++;
    }
}

// E) does assigning a value already held in a constant register avoid the
//    SFPLOADI + liveness SFPMOV pair? (It does not -- same 8 instr/row as the
//    plain literal form in 01. vConst* is deprecated; naming it explicitly is
//    the point of this probe, so the warning is suppressed here only.)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" void e_eqz_const_reg() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = vConst0;
        v_if(abs_bits == 0) { result = vConst1; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}
#pragma GCC diagnostic pop

// F) predicated store + addr_mode fold together: closest sfpi can get to the
//    raw-TTI eqz shape.
extern "C" void f_eqz_best_effort() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits == 0) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}

// G) fp32 strict-ordered lt, best effort: hoisted inf + predicated store + addr_mode.
extern "C" void g_fp32_lt_best_effort(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = FP32_INF_BITS;
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
