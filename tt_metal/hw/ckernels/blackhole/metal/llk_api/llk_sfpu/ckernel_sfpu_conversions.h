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

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
//
// Reference form: 8 SFPU instructions per call, three of them SFPLOADI that the compiler
// re-materialises in every row of a loop. Inside a row loop use bf16_rne_bias() +
// float32_to_bf16_rne_for_store() below, which is the same algorithm at 4 instructions.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32)
    // This is needed for the tie-breaker: round to even
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow → rounds up
    // - If lower 16 bits < 0x8000: no overflow → rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: 0x7fff+0=0xffff, no overflow → stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: 0x7fff+1=0x8000, overflow → rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32)
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float
    return sfpi::as<sfpi::vFloat>(bits);
}

// The 0x7fff addend of the RNE algorithm, to be created once before a row loop and passed to
// float32_to_bf16_rne_for_store(). Kept in a vector register across the loop, it costs one
// SFPLOADI per call instead of one per row.
sfpi_inline sfpi::vUInt bf16_rne_bias() { return sfpi::vUInt(0x7fffU); }

// fp32 -> bf16 round-to-nearest-even for a value that is about to be stored to a 16-bit Dest.
//
// Same "bits + 0x7fff + lsb" algorithm as float32_to_bf16_rne(), so the high 16 bits are
// bit-identical to it for every input (ties to even, carry into the exponent up to +/-inf,
// denormals, NaN payloads), at 4 SFPU instructions per row instead of 8:
//   - the addend comes in as `bias` (from bf16_rne_bias(), hoisted out of the loop) instead of
//     being re-materialised by SFPLOADI in every row;
//   - the bf16 LSB is extracted with two immediate shifts, (bits << 15) >> 31, instead of a
//     shift and an AND against a materialised 1;
//   - the low 16 bits are NOT cleared. They are unspecified in the returned value: the caller
//     must store it to a 16-bit (Float16_b) Dest, whose SFPSTORE keeps only the high half.
// Do not use it for a value that stays in fp32 or that is read back before the store.
//
// The hardware alternative, SFPSTOCHRND FP32_TO_FP16B in "nearest" mode, was measured on
// Blackhole silicon (2026-09-28, 4096 patterns) and is not a substitute: it rounds ties away
// from zero (bf16-even lanes at exactly half a ULP round up), maps every NaN to +/-inf and
// flushes denormals to +0.
sfpi_inline sfpi::vFloat float32_to_bf16_rne_for_store(sfpi::vFloat in, const sfpi::vUInt bias) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    // Bit 16 of the fp32 pattern is the bf16 LSB; it breaks the tie towards even.
    sfpi::vUInt lsb = (bits << 15) >> 31;
    return sfpi::as<sfpi::vFloat>(bits + bias + lsb);
}
