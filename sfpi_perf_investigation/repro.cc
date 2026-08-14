// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for the four claims that carry the argument in ISSUE.md.
// The 0*.cc probes cover the full matrix; start here.
#include <cstdint>
#include "shim.h"

using namespace sfpi;

constexpr int AM6 = 6;  // metal ADDR_MOD_6: dest.incr = 2

// Idiomatic. This is the shape PR #52932 shipped.  -> 14 instr/row
extern "C" void ltz_idiomatic() {
    vInt inf_bits = 0x7F800000;
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

// Same algorithm with all four workarounds applied: predicated store (ISSUE 5.2),
// addr_mode fold (5.3), nested v_if (5.4), commuted compare (5.1).
// -> 8 instr/row, i.e. exact parity with the raw TTI this replaces.
extern "C" void ltz_tuned() {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(v < 0.0f) {
            v_if(abs_bits != 0) {
                v_if(inf_bits >= abs_bits) { dst_reg[0].mode<DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// ISSUE 5.6: emits a bare `SFPIADD L0, L1, 0, 2`, a two's-complement subtract
// with the condition code taken from the sign, so INT32_MAX > -1 answers false.
extern "C" void int32_lt_wrong(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * 32].mode<DataLayout::I32>();
        vInt b = dst_reg[i1 * 32].mode<DataLayout::I32>();
        vInt r = 0;
        v_if(a < b) { r = 1; }
        v_endif;
        dst_reg[io * 32].mode<DataLayout::I32>() = r;
        dst_reg++;
    }
}

// ISSUE 5.5: SFPGT is reachable as a builtin and does emit `SFPGT L0, L1, 0, 8`,
// but only as a 0/-1 lane mask. sfpi::vBool cannot be built from it, so v_if can
// never consume a total-order compare.
extern "C" void sfpgt_reachable(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * 32];
        vFloat b = dst_reg[i1 * 32];
        vUInt mask = as<vUInt>(vFloat(__builtin_rvtt_sfpgt(a.get(), b.get(), 8)));
        dst_reg[io * 32].mode<DataLayout::U32>() = mask >> 31;
        dst_reg++;
    }
}
