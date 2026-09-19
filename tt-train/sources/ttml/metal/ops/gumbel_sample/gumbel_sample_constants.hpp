// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Constants of the approximate -ln(v) in the Gumbel noise chain, shared between device and host
// (the const_utils.hpp pattern: a plain header both sides can compile). The TRISC side
// (device/kernels/compute/gumbel_sfpu.h) programs three of them into the SFPU's programmable const
// registers and inlines B as an fp16a immediate; the host side (TestGumbelApproxLogInvariants in
// trivial_ttnn_ops_test.cpp) reconstructs the polynomial from these SAME values and pins the
// monotonicity, endpoint-tie and error-bound invariants the kernel relies on -- so a re-fit cannot
// drift the kernel and its proof apart.
//
// Constraints when re-fitting (full derivation and invariants in gumbel_sfpu.h):
//  - kGumbelPolyB must stay exactly representable on the fp16a grid (it loads as a single-SFPLOADI
//    inline immediate on the device);
//  - kGumbelPolyC and kGumbelPolyD must stay EXACT derivations from ln2 and B as written below, or
//    the octave-boundary ties the host test asserts break.

namespace ttml::metal::sfpu {

constexpr float kGumbelNegLn2 = -0x1.62e43p-1F;  // -ln(2), full fp32 -- lives in a Prgm reg
constexpr float kGumbelPolyB = 0.240234375F;     // fp16a-exact minimax under the ties (inline immediate)
constexpr float kGumbelPolyC = -0x1.69f218p+0F;  // kGumbelNegLn2 - 3*kGumbelPolyB, fp32-exact
constexpr float kGumbelPolyD = 0x1.2c7228p+0F;   // 2*kGumbelPolyB - kGumbelNegLn2 + 2^-20, fp32-exact

}  // namespace ttml::metal::sfpu
