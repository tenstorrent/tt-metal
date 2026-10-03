// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Helper function for _sfpu_binary_power_
// This function is based on _float32_to_int32_, but expects a positive input, which simplifies the code
// and makes it faster
sfpi_inline sfpi::vInt _float_to_int32_positive_(sfpi::vFloat in) {
    sfpi::vInt result;
    sfpi::vInt exp = exexp(in);  // extract exponent
    v_if(exp < 0) { result = 0; }
    v_elseif(exp > 30)  // overflow occurs above this range
    {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<int32_t>::max();
    }
    v_else {
        // extract mantissa
        sfpi::vInt man = exman(in, sfpi::MantissaMode::ImplicitOne);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23;  // 23 is number of mantissa bits in float32
        man = shft(man, shift, sfpi::ShiftMode::Logical);

        result = man;
    }
    v_endif;
    return result;
}

// The 0x7fff addend of float32_to_bf16_rne_for_store(). Materialise it once, before the row
// loop: sfpi re-issues an SFPLOADI per row for a literal used inside the loop. Binding it
// unconditionally is free for an instantiation that never rounds: the compiler drops a pre-loop
// constant nothing reads (every non-rounding arm of the six call sites is instruction-identical
// to a build without it, sfpi 7.83.0 and 7.84.0).
sfpi_inline sfpi::vUInt bf16_rne_bias() { return sfpi::vUInt(0x7fffU); }

// fp32 -> bf16 round-to-nearest-even, 4 SFPU instructions: bits + 0x7fff + lsb, where lsb is
// the bf16 LSB (fp32 bit 16), so a tie (low half exactly 0x8000) carries only when that LSB is
// 1 and rounds to even. A carry out of the mantissa bumps the exponent; only the largest finite
// binade rounds up to +/-inf. Canonical NaN (0x7FC00000, what SFPMAD emits) and a bf16-sourced
// NaN (zero low half) stay NaN; other fp32 NaN payloads are not preserved (0x7F800001 rounds to
// +inf, 0x7FFFFFFF wraps to -0); none of the call sites can produce one.
// The low 16 bits are left unspecified: store the result straight to a 16-bit Float16_b Dest,
// whose SFPSTORE keeps only the high half. Never read it back or store it as fp32.
// SFPSTOCHRND is not a substitute: on Blackhole silicon it rounds ties away from zero, maps
// NaN to +/-inf and flushes denormals to +0.
sfpi_inline sfpi::vFloat float32_to_bf16_rne_for_store(sfpi::vFloat in, const sfpi::vUInt bias) {
    constexpr int bf16_lsb_bit = 16;
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits << (31 - bf16_lsb_bit)) >> 31;
    return sfpi::as<sfpi::vFloat>(bits + bias + lsb);
}
