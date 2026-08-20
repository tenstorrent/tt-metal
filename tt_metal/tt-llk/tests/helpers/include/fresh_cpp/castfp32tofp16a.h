// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `castfp32tofp16a` corpus row (metal
// cast_fp32_to_fp16a), re-specced by lane CX (2026-08-20, owner-signed).
//
// Semantic definition = the row's golden = the HARDWARE cast semantics of
// SFP_STOCH_RND mod1=FP32_TO_FP16A rnd_mode=0, machine-checked by lane CT's
// exhaustive 2^32 proof against the pinned craq oracle (sfpi-gcc
// agent/cast-peephole-harvest: gcc/config/riscv/tt/proofs/cast-fp16a-rne/)
// and independently documented by the ISA reference (tt-isa-documentation
// BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md —
// "Round to nearest with ties away from zero", denormals/-0.0 -> +0.0,
// +/-NaN -> +/-Infinity; its SFPSTORE table also confirms the fp32 store of
// the rounded value is exact):
// half-AWAY rounding on the 13 discarded mantissa bits (>= midpoint rounds
// up, despite the rnd encoding being NAMED RND_EVEN), exponent-0 inputs
// (zeros and denormals) flushed to +0.0, every NaN payload collapsed to
// signed infinity, exponent kept fp32-range (a mantissa carry may ripple
// into the exponent, saturating max normals to signed infinity).
//
// The typed statement of that function IS the hardware convert —
// sfpi::convert<vFloat16a>(x, RoundMode::Nearest) (sfpi_lib.h; on WH/BH
// RoundMode::Nearest aliases NearestAway) — so the semantic body states it
// directly.  The previous software-RNE body (preserved verbatim in
// castfp32tofp16a_legacy.h) implemented the OLD golden's model, which CT's
// proof showed disagrees with the hardware on 33,810,429 of 2^32 inputs;
// under the re-specced golden it fails at exactly those classes.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_cast_fp32_to_fp16a_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Assign the vFloat16a convert result back through a vFloat before the
        // store (production form): storing the vFloat16a-typed value directly
        // emits SFPSTORE mod0=1 (a 16A-format store) — wrong for the fp32/
        // dest_acc row contract and unimplemented in the pinned BH sim.
        v                = sfpi::convert<sfpi::vFloat16a>(v, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
