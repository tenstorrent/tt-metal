// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Helper function for _sfpu_binary_power_
// This function is based on _float32_to_int32_, but expects a positive input, which simplifies the code
// and makes it faster
sfpi_inline sfpi::vInt _float_to_int32_positive_(sfpi::vFloat in) {
    sfpi::vInt exp = sfpi::exexp(in);  // extract exponent
    // In-range value, computed for every lane: the mantissa with its implicit one, shifted left by
    // (exponent - 23) (23 is the number of float32 mantissa bits). SFPSHFT has no side effects, so lanes that
    // are out of range (|in| < 1.0 shifts right by up to 150; exp >= 31 overflows) just produce a value the
    // fix-ups below overwrite. Keeping it out of the predicate avoids the PUSHC/POPC and register shuffles of
    // a three-arm v_if / v_elseif / v_else chain.
    sfpi::vInt man = sfpi::exman(in, sfpi::MantissaMode::ImplicitOne);
    sfpi::vInt result = sfpi::shft(man, exp - 23, sfpi::ShiftMode::Logical);
    v_if(exp < 0) { result = 0; }                                          // |in| < 1.0 (incl. zero and denormals)
    v_elseif(exp >= 31) { result = std::numeric_limits<int32_t>::max(); }  // overflow: saturate to INT32_MAX
    v_endif;
    return result;
}

// Float32 -> bfloat16 IEEE 754 Round-to-Nearest-Even (RNE): the "add 0x7fff + LSB" algorithm.
// kBf16RneBias is the round-half-down bias, kBf16RneMask keeps the 16 bits a bf16 holds, and
// kBf16LsbBit is the float32 bit that becomes the bf16 mantissa LSB (the tie-breaker).
constexpr std::uint32_t kBf16RneBias = 0x7fffU;
constexpr std::uint32_t kBf16RneMask = 0xFFFF0000U;
constexpr std::uint32_t kBf16LsbBit = 16;

// RNE with the two constants bound by the caller. sfpi re-materialises an integer literal
// (SFPLOADI) at every use, so a caller that binds kBf16RneBias and kBf16RneMask once before
// its row loop keeps them in two LREGs for the whole face instead of reloading them per row.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in, const sfpi::vUInt bias, const sfpi::vUInt mask) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);

    // The tie-breaker: the LSB of what will become the bf16 mantissa, isolated by a logical
    // shift pair, which takes immediates and so needs no constant LREG (a mask would).
    sfpi::vUInt lsb = (bits << (31 - kBf16LsbBit)) >> 31;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow → rounds up
    // - If lower 16 bits < 0x8000: no overflow → rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: 0x7fff+0=0xffff, no overflow → stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: 0x7fff+1=0x8000, overflow → rounds up to even
    bits = bits + bias + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32)
    bits = bits & mask;

    // Reinterpret back as float
    return sfpi::as<sfpi::vFloat>(bits);
}

// RNE with the constants loaded at the call. For callers that cannot spare two LREGs across
// their loop (addcdiv, binary_div: all 8 are live inside the reciprocal).
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    return float32_to_bf16_rne(in, kBf16RneBias, kBf16RneMask);
}
