// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Constants of the approximate -ln(v) in the Gumbel noise chain, compiled by BOTH the TRISC pass
// (gumbel_sfpu.h) and the host test that pins its invariants (TestGumbelApproxLogInvariants), so a
// re-fit cannot drift the kernel and its proof apart. Re-fitting constraints: B must stay on the
// fp16a grid (it loads as a single inline immediate); C and D must stay EXACT derivations from ln2
// and B as written, or the octave-boundary ties the host test asserts break.

namespace ttml::metal::sfpu {

constexpr float kGumbelNegLn2 = -0x1.62e43p-1F;  // -ln(2), full fp32 -- lives in a Prgm reg
constexpr float kGumbelPolyB = 0.240234375F;     // fp16a-exact minimax under the ties (inline immediate)
constexpr float kGumbelPolyC = -0x1.69f218p+0F;  // kGumbelNegLn2 - 3*kGumbelPolyB, fp32-exact
constexpr float kGumbelPolyD = 0x1.2c7228p+0F;   // 2*kGumbelPolyB - kGumbelNegLn2 + 2^-20, fp32-exact

}  // namespace ttml::metal::sfpu
