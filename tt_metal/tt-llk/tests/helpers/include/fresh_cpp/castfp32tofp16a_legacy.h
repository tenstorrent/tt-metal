// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// castfp32tofp16a — LEGACY semantic body, preserved verbatim (symbol renamed
// *_legacy) when lane CX re-specced the row's golden to the hardware cast
// semantics (2026-08-20, owner-signed) and replaced the live body with the
// production form sfpi::convert<vFloat16a>.  Kept for A/B archaeology; not
// wired to any test node.  The live body is fresh_cpp/castfp32tofp16a.h.
//
// Why it was retired: this body implements round-half-to-EVEN with
// inf/NaN passthrough — the OLD golden's model — which lane CT's exhaustive
// 2^32 sweep against the pinned craq oracle proved is NOT what the hardware
// cast (SFP_STOCH_RND FP32_TO_FP16A rnd_mode=0) computes: the hardware rounds
// half-AWAY (despite the encoding being named RND_EVEN), flushes exponent-0
// inputs to +0.0, and collapses every NaN to signed infinity — 33,810,429
// mismatches in four exact classes (sfpi-gcc agent/cast-peephole-harvest:
// gcc/config/riscv/tt/proofs/cast-fp16a-rne/).  Under the re-specced golden
// this body FAILS at exactly those classes on the rounding-visible pipeline
// (lane CX's archived differential run) — that failing run is the proof the
// new golden actually observes the rounding.  It also cost 18 issue words per
// row against the hardware convert's 3 (the board's +378% loss, closed by the
// re-spec with zero compiler change).
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_cast_fp32_to_fp16a_fresh_cpp_legacy()
{
    constexpr int EXPONENT_FIELD = 0x7F800000;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        const sfpi::vUInt u  = sfpi::as<sfpi::vUInt>(v);

        // Round-half-to-even on the dropped 13 mantissa bits: adding
        // (halfway - 1) + kept-LSB rounds up exactly when the remainder is
        // above halfway, or equal to halfway with an odd kept bit.
        const sfpi::vUInt kept_lsb = (u >> 13) & 1;
        sfpi::vUInt rounded        = u + 0x0FFF + kept_lsb;
        rounded                    = rounded & 0xFFFFE000u;
        sfpi::vFloat r             = sfpi::as<sfpi::vFloat>(rounded);

        // Non-finite input (inf/nan): pass the bit pattern through unchanged.
        const sfpi::vInt exponent_bits = sfpi::as<sfpi::vInt>(v) & EXPONENT_FIELD;
        v_if (exponent_bits == EXPONENT_FIELD)
        {
            r = v;
        }
        v_endif;

        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
